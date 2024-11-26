# -*- coding: utf-8 -*-
import re
import time
import logging
import os
import io
from threading import Lock

from zipfile import ZipFile, is_zipfile
from urllib.parse import urljoin
from requests import Session

from babelfish import language_converters
from subzero.language import Language
from subliminal import Episode, Movie
from subliminal.exceptions import ConfigurationError, ProviderError, DownloadLimitExceeded
from subliminal_patch.exceptions import APIThrottled
from .mixins import ProviderRetryMixin
from subliminal_patch.subtitle import Subtitle
from subliminal.subtitle import fix_line_ending
from subliminal_patch.providers import Provider
from subliminal_patch.providers import utils

logger = logging.getLogger(__name__)

retry_amount = 3
retry_timeout = 5

language_converters.register('subdl = subliminal_patch.converters.subdl:SubdlConverter')


class SubdlSubtitle(Subtitle):
    provider_name = 'subdl'
    hash_verifiable = False
    hearing_impaired_verifiable = True

    def __init__(self, language, forced, hearing_impaired, page_link, download_link, file_id, release_names, uploader,
                 season=None, episode=None, episode_from=None, episode_to=None, full_season=False):
        super().__init__(language)
        language = Language.rebuild(language, hi=hearing_impaired, forced=forced)

        self.season = season
        self.episode = episode
        self.episode_from = episode_from
        self.episode_to = episode_to
        self.full_season = full_season  # Use the API-provided flag
        self.releases = release_names
        self.release_info = ', '.join(release_names) if release_names else ''
        self.language = language
        self.forced = forced
        self.hearing_impaired = hearing_impaired
        self.file_id = file_id
        self.page_link = page_link
        self.download_link = download_link
        self.uploader = uploader
        self.matches = None

    @property
    def id(self):
        return self.file_id

    def get_matches(self, video):
        matches = set()

        # handle movies and series separately
        if isinstance(video, Episode):
            # series
            matches.add('series')
            
            # season
            if video.season == self.season:
                matches.add('season')
            logger.debug(self.full_season)
            # episode matching (only if not a season pack)
            if not self.full_season:
                if video.episode == self.episode:
                    matches.add('episode')
            else:
                
                matches.add('episode')
            # imdb
            matches.add('series_imdb_id')
        else:
            # title
            matches.add('title')
            
            # imdb
            matches.add('imdb_id')

        utils.update_matches(matches, video, self.release_info)

        self.matches = matches
        return matches



class SubdlProvider(ProviderRetryMixin, Provider):
    """Subdl Provider"""
    server_hostname = 'api.subdl.com'

    languages = {Language(*lang) for lang in list(language_converters['subdl'].to_subdl.keys())}
    languages.update(set(Language.rebuild(lang, forced=True) for lang in languages))
    languages.update(set(Language.rebuild(l, hi=True) for l in languages))

    video_types = (Episode, Movie)

    def __init__(self, api_key=None):
        if not api_key:
            raise ConfigurationError('Api_key must be specified')

        self.session = Session()
        self.session.headers = {'User-Agent': os.environ.get("SZ_USER_AGENT", "Sub-Zero/2")}
        self.api_key = api_key
        self.video = None
        self._started = None
        self._season_pack_cache = {}
        self._search_cache = {}
        self._cache_timestamp = {}
        self._cache_duration = 900  # -----------------------------> Cache duration (15m)
        self._max_cache_size = 100  # items
        self._cache_lock = Lock()  # Add thread safety



    def initialize(self):
        self._started = time.time()

    def terminate(self):
        self.session.close()

    def server_url(self):
        return f'https://{self.server_hostname}/api/v1/'

    def query(self, languages, video):
        self.video = video
        
        if isinstance(self.video, Episode):
            title = self.video.series
        else:
            title = self.video.title
            
        imdb_id = None
        if isinstance(self.video, Episode) and self.video.series_imdb_id:
            imdb_id = self.video.series_imdb_id
        elif isinstance(self.video, Movie) and self.video.imdb_id:
            imdb_id = self.video.imdb_id

        # be sure to remove duplicates using list(set())
        langs_list = sorted(list(set([language_converters['subdl'].convert(lang.alpha3, lang.country, lang.script) for
                                      lang in languages])))

        langs = ','.join(langs_list)
        logger.debug(f'Searching for those languages: {langs}')
        search_cache_key = self._search_cache_key(video, langs)
        cached_results = self._get_from_cache(search_cache_key, cache_type='search')
        if cached_results is not None:
            logger.debug('Using cached search results')
            return cached_results
        # query the server
        if isinstance(self.video, Episode):
            res = self.retry(
                lambda: self.session.get(self.server_url() + 'subtitles',
                                         params=(('api_key', self.api_key),
                                                 ('episode_number', self.video.episode),
                                                 ('film_name', title if not imdb_id else None),
                                                 ('imdb_id', imdb_id if imdb_id else None),
                                                 ('languages', langs),
                                                 ('season_number', self.video.season),
                                                 ('subs_per_page', 30),
                                                 ('type', 'tv'),
                                                 ('comment', 1),
                                                 ('releases', 1),
                                                 ('bazarr', 1)),  # this argument filter incompatible image based or
                                         # txt subtitles
                                         timeout=30),
                amount=retry_amount,
                retry_timeout=retry_timeout
            )
        else:
            res = self.retry(
                lambda: self.session.get(self.server_url() + 'subtitles',
                                         params=(('api_key', self.api_key),
                                                 ('film_name', title if not imdb_id else None),
                                                 ('imdb_id', imdb_id if imdb_id else None),
                                                 ('languages', langs),
                                                 ('subs_per_page', 30),
                                                 ('type', 'movie'),
                                                 ('comment', 1),
                                                 ('releases', 1),
                                                 ('bazarr', 1)),  # this argument filter incompatible image based or
                                         # txt subtitles
                                         timeout=30),
                amount=retry_amount,
                retry_timeout=retry_timeout
            )

        if res.status_code == 429:
            raise APIThrottled("Too many requests")
        elif res.status_code == 403:
            raise ConfigurationError("Invalid API key")
        elif res.status_code != 200:
            res.raise_for_status()

        subtitles = []

        result = res.json()

        if ('success' in result and not result['success']) or ('status' in result and not result['status']):
            logger.debug(result["error"])
            return []

        logger.debug(f"Query returned {len(result['subtitles'])} subtitles")

        if len(result['subtitles']):
            for item in result['subtitles']:
                is_season_pack = self._is_season_pack(item)


                subtitle = SubdlSubtitle(
                        language=Language.fromsubdl(item['language']),
                        forced=self._is_forced(item),
                        hearing_impaired=item.get('hi', False) or self._is_hi(item),
                        page_link=urljoin("https://subdl.com", item.get('subtitlePage', '')),
                        download_link=item['url'],
                        file_id=item['name'],
                        release_names=item.get('releases', []),
                        uploader=item.get('author', ''),
                        season=item.get('season', None),
                        episode=item.get('episode', None) ,
                        episode_from=item.get('episode_from', None),
                        episode_to=item.get('episode_end', None),
                        full_season=is_season_pack  # Directly use the value from the result
                        )
                    
                subtitle.get_matches(self.video)
                if subtitle.language in languages:  # make sure only desired subtitles variants are returned
                    subtitles.append(subtitle)
            self._store_in_cache(search_cache_key, subtitles, 'search')

        return subtitles

    @staticmethod
    def _is_hi(item):
        # Comments include specific mention of removed or non HI
        non_hi_tag = ['hi remove', 'non hi', 'nonhi', 'non-hi', 'non-sdh', 'non sdh', 'nonsdh', 'sdh remove']
        for tag in non_hi_tag:
            if tag in item.get('comment', '').lower():
                return False

        # Archive filename include _HI_
        if '_hi_' in item.get('name', '').lower():
            return True

        # Comments or release names include some specific strings
        hi_keys = [item.get('comment', '').lower(), [x.lower() for x in item.get('releases', [])]]
        hi_tag = ['_hi_', ' hi ', '.hi.', 'hi ', ' hi', 'sdh', '𝓢𝓓𝓗']
        for key in hi_keys:
            if any(x in key for x in hi_tag):
                return True

        # nothing match so we consider it as non-HI
        return False

    @staticmethod
    def _is_forced(item):
        # Comments include specific mention of forced subtitles
        forced_tags = ['forced', 'foreign']
        for tag in forced_tags:
            if tag in item.get('comment', '').lower():
                return True

        # nothing match so we consider it as normal subtitles
        return False
    def _cache_key(self, video, language):
        """Generate a unique cache key for a season pack or search."""
        if isinstance(video, Episode):
            return f"{video.series}_{video.season}_{language}"
        elif isinstance(video, Movie):
            return f"{video.title}_{video.year}_{language}"
        return None

    def _search_cache_key(self, video, language):
        """Generate a unique cache key for search results."""
        if isinstance(video, Episode):
            return f"search_{video.series}_{video.season}_{language}"
        elif isinstance(video, Movie):
            return f"search_{video.title}_{video.year}_{language}"
        return None

    def _is_cache_valid(self, cache_key):
        """Check if the cache entry is still valid."""
        if cache_key in self._cache_timestamp:
            return (time.time() - self._cache_timestamp[cache_key]) < self._cache_duration
        return False

    def _store_in_cache(self, cache_key: str, data: any, data_type: str = 'pack') -> None:
        """Store data in cache with size limit enforcement.
        
        Args:
            cache_key: Unique identifier for cached item
            data: Data to cache
            data_type: Type of cache to use ('pack' or 'search')"""
        with self._cache_lock:
            cache = self._season_pack_cache if data_type == 'pack' else self._search_cache
            
            # Implement simple LRU (Least Recently Used) by removing oldest items if cache is full
            if len(cache) >= self._max_cache_size:
                oldest_key = min(self._cache_timestamp, key=self._cache_timestamp.get)
                cache.pop(oldest_key, None)
                self._cache_timestamp.pop(oldest_key, None)
            
            cache[cache_key] = data
            self._cache_timestamp[cache_key] = time.time()

    def _get_from_cache(self, cache_key, episode_num=None, cache_type='pack'):
        """Get data from appropriate cache."""
        if cache_type == 'pack':
            if cache_key in self._season_pack_cache and self._is_cache_valid(cache_key):
                cached_data = self._season_pack_cache[cache_key]
                archive_stream = io.BytesIO(cached_data['archive_content'])
                
                if is_zipfile(archive_stream):
                    with ZipFile(archive_stream) as archive:
                        subtitle = cached_data['subtitle_info']
                        subtitle.episode = episode_num
                        return self._handle_season_pack(archive, subtitle)
        elif cache_type == 'search':
            if cache_key in self._search_cache and self._is_cache_valid(cache_key):
                return self._search_cache[cache_key]
        return None
    def _cleanup_cache(self):
        """Remove expired cache entries."""
        current_time = time.time()
        with self._cache_lock:
            expired_keys = [
                key for key, timestamp in self._cache_timestamp.items()
                if current_time - timestamp > self._cache_duration
            ]
            for key in expired_keys:
                self._season_pack_cache.pop(key, None)
                self._search_cache.pop(key, None)
                self._cache_timestamp.pop(key, None)
                
    def _is_season_pack(self, data):
        """ Determines if the given subtitle data corresponds to a full season pack.

        Args:
            data (dict): Dictionary containing subtitle metadata.

        Returns:
            bool: True if the subtitle is for a full season pack, False otherwise."""
        release_name = data.get("release_name", "").lower()
        comment = data.get("comment", "").lower()
        episode_from = data.get("episode_from")
        episode_end = data.get("episode_end")
        full_season = data.get("full_season", False)
        
        # Regular expressions for season detection
        season_pattern = re.compile(r'\bS\d{2}\b', re.IGNORECASE)  # Matches S00, S01, etc.
        season_pack_keywords = ["full season", "season pack", "complete"," كامل", ]
        partial_keywords = ["part 1", "part 2", "volume"]

        # Conditions
        conditions_met = 0
        
        # Condition 1: best case scenario
        if full_season and episode_from is not None and episode_end is not None and episode_from != episode_end:
            return True

        # Condition 2: Release name or comment contains season pack indicators
        if any(keyword in release_name for keyword in season_pack_keywords) or any(keyword in comment for keyword in season_pack_keywords):
            conditions_met += 1

        # Condition 3: Season pattern exists in release name but no episode-specific pattern
        if season_pattern.search(release_name) and not re.search(r'\bS\d{2}E\d{2}\b', release_name, re.IGNORECASE):
            conditions_met += 1

        # Condition 4: Episode range indicates multiple episodes
        if episode_from is not None and episode_end is not None and episode_from < episode_end:
            conditions_met += 1

        # Condition 5: Keywords like "part 1", "volume" are absent in release name or comment
        if not any(keyword in release_name for keyword in partial_keywords) and not any(keyword in comment for keyword in partial_keywords):
            conditions_met += 1

        # Return True if at least 2 conditions are met
        return conditions_met >= 2
    def list_subtitles(self, video, languages):
        return self.query(languages, video)

    def download_subtitle(self, subtitle):
        cache_key = self._cache_key(self.video, subtitle.language)
        if cache_key and isinstance(self.video, Episode):
            cached_result = self._get_from_cache(cache_key, self.video.episode)
            if cached_result:
                logger.debug('Using cached season pack subtitle')
                subtitle.content = cached_result
                return
        logger.debug('Downloading subtitle %r', subtitle)
        download_link = urljoin("https://dl.subdl.com", subtitle.download_link)

        r = self.retry(
            lambda: self.session.get(download_link, timeout=30),
            amount=retry_amount,
            retry_timeout=retry_timeout
        )

        if r.status_code == 429:
            raise DownloadLimitExceeded("Daily download limit exceeded")
        elif r.status_code == 403:
            raise ConfigurationError("Invalid API key")
        elif r.status_code != 200:
            r.raise_for_status()

        if not r:
            logger.error(f'Could not download subtitle from {download_link}')
            subtitle.content = None
            return

        
        archive_stream = io.BytesIO(r.content)
        if is_zipfile(archive_stream):
            archive = ZipFile(archive_stream)
            if subtitle.full_season and isinstance(self.video, Episode) and cache_key:
                self._store_in_cache(cache_key, r.content, subtitle)
            if subtitle.full_season and isinstance(self.video, Episode):
                return self._handle_season_pack(archive, subtitle)
            else:
                return self._handle_single_subtitle(archive, subtitle)
        else:
            logger.error(f'Could not unzip subtitle from {download_link}')
            subtitle.content = None

    def _handle_season_pack(self, archive, subtitle):
        """Handle season pack archives and extract the correct episode."""
        subtitle_files = []
        episode_pattern = None
        
        if isinstance(self.video, Episode):
            episode_pattern = [
                f'e{self.video.episode:02d}',
                f'ep{self.video.episode:02d}',
                f'episode{self.video.episode:02d}',
                f'episode {self.video.episode:02d}',
                f'{self.video.episode:02d}'
            ]

        for name in archive.namelist():
            if not name.endswith(('srt', 'ssa', 'ass', 'sub')):
                continue

            file_info = archive.getinfo(name)
            score = 0
            
            # Score based on filename matching
            name_lower = name.lower()
            
            # Episode number matching
            if episode_pattern:
                if any(pattern in name_lower for pattern in episode_pattern):
                    score += 1

            # Season matching
            if f's{self.video.season:02d}' in name_lower:
                score += 2

            # Release name matching
            if subtitle.releases:
                for release in subtitle.releases:
                    if release.lower() in name_lower:
                        score += 3
                        break

            subtitle_files.append({
                'name': name,
                'size': file_info.file_size,
                'content': archive.read(name),
                'score': score
            })

        if not subtitle_files:
            logger.error('No valid subtitle files found in season pack')
            subtitle.content = None
            return

        # Sort by score and get the best match
        subtitle_files.sort(key=lambda x: x['score'], reverse=True)
        best_match = subtitle_files[0]

        if best_match['score'] > 2:
            logger.debug(f'Selected {best_match["name"]} from season pack with score {best_match["score"]}')
            subtitle.content = fix_line_ending(best_match['content'])
        else:
            logger.error('No suitable subtitle found in season pack')
            subtitle.content = None

    def _handle_single_subtitle(self, archive, subtitle):
        """Handle single subtitle archives."""
        subtitle_files = []
        
        for name in archive.namelist():
            if name.endswith(('srt', 'ssa', 'ass', 'sub')):
                file_info = archive.getinfo(name)
                subtitle_files.append({
                    'name': name,
                    'size': file_info.file_size,
                    'content': archive.read(name)
                })

        if not subtitle_files:
            logger.error('No valid subtitle files found')
            subtitle.content = None
            return

        if len(subtitle_files) == 1:
            subtitle.content = fix_line_ending(subtitle_files[0]['content'])
            return

        # Multiple files - select best match
        chosen_subtitle = self._select_best_subtitle(subtitle_files, subtitle)
        if chosen_subtitle:
            subtitle.content = fix_line_ending(chosen_subtitle['content'])
        else:
            subtitle.content = None
