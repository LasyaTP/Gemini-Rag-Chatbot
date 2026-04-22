# Environment Setup Guide

## Setting up your API key

1. **Create a `.env` file** in the project root directory (same folder as `app.py`)

2. **Add your Google API key** to the `.env` file:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   USE_LOCAL_EMBEDDINGS=true
   ```

3. **Get your API key** from: https://makersuite.google.com/app/apikey

## Embedding Options

### Local Embeddings (Recommended - No Quota Limits)
By default, the app uses **local HuggingFace embeddings** which:
- ✅ **No quota limits** - works completely free
- ✅ **Works offline** - no API calls needed
- ✅ **Fast and reliable** - runs on your machine
- ✅ **No billing concerns**

To use local embeddings (default), add to your `.env`:
```
USE_LOCAL_EMBEDDINGS=true
```

### Google Embeddings (Requires API Quota)
If you want to use Google's embedding API instead:
- ⚠️ **Has quota limits** on free tier
- ⚠️ **Requires API calls** - needs internet
- ⚠️ **May hit rate limits** on free tier

To use Google embeddings, add to your `.env`:
```
USE_LOCAL_EMBEDDINGS=false
```

**Note:** Even with local embeddings, you still need `GOOGLE_API_KEY` for the LLM (Gemini) to generate responses.

## Important Notes

- The `.env` file is already added to `.gitignore` so it won't be committed to git
- Never share your API key publicly
- The `.env` file will be automatically loaded when you run the app

## Example `.env` file content:

```
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
USE_LOCAL_EMBEDDINGS=true
```

Replace `AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` with your actual API key.


