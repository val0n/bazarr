
## Subdl Provider for Bazarr

This is an updated version of the `subdl.py` file for the **Bazarr** project. It introduces additional functionality and optimizations to improve subtitle handling, particularly for full-season packs, caching, and error handling during API queries.

---

### Features

#### 1. Full-Season Subtitle Support
- Adds support for full-season subtitle packs.
- Extracts specific episodes from season packs using:
  - Episode numbers.
  - Season numbers.
  - Release name matches.
- Caches season packs to avoid repeated downloads.

#### 2. Caching System
- Implements caching for:
  - **Season packs**: Stores downloaded full-season subtitles for reuse.
  - **Search results**: Reduces API calls by saving query responses temporarily.
- Caching is time-limited (default: 15 minutes) and size-controlled for optimal performance.

#### 3. Better API Query Handling
- Adds support for additional API parameters.

#### 4. Improved Subtitle Matching
- Matches subtitles more accurately to videos using metadata.

---

### Installation

1. **Download the `subdl.py` File**:
   - Clone this repository or download the **[subdl.py file directly](https://github.com/val0n/bazarr/blob/master/custom_libs/subliminal_patch/providers/subdl.py)**.
2. **Replace the Existing File**:
   - Navigate to your Bazarr installation directory.
   - Locate the `custom_libs/subliminal_patch/providers/` folder.
   - Replace the existing `subdl.py` file with this version.
3. **Restart Bazarr**:
   - Restart your Bazarr service to apply the changes.
