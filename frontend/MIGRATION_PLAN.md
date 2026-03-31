# Frontend Unification Plan (Vercel Prep)

## Target Structure

frontend/
  package.json
  vite.config.js
  .env.example
  src/
    main.jsx
    App.jsx
    routes/
      AppRoutes.jsx
    pages/
      LoginPage.jsx
      DashboardPage.jsx
      UnifiedEmotionPage.jsx
    components/
      auth/
        Login.jsx
      dashboard/
        Dashboard.jsx
      emotion/
        CameraEmotionDetection.jsx
        SpeechEmotionDetection.jsx
        UnifiedEmotionDetection.jsx
      common/
        LoadingSpinner.jsx
        ErrorBanner.jsx
    services/
      apiClient.js
      emotionApi.js
    hooks/
      useCameraEmotion.js
      useSpeechEmotion.js
    styles/
      index.css

## Migration Steps

1. Establish a single frontend source of truth in frontend/src and stop adding feature work under FYP_frontend/FYP_deployment.
2. Move UnifiedEmotionDetection.jsx into frontend/src/components/emotion and convert to use the same API service layer as CameraEmotionDetection and SpeechEmotionDetection.
3. Introduce react-router-dom and create AppRoutes.jsx with routes for login, dashboard, and unified emotion page.
4. Simplify App.jsx to render AppRoutes only; keep auth state in a context or route guard.
5. Extract API base URL and endpoint calls into services/apiClient.js and services/emotionApi.js.
6. Replace duplicate UI components between the two trees with a single implementation in frontend/src/components.
7. Add .env.example with VITE_API_BASE_URL and configure Vercel environment variables.
8. Remove or archive FYP_frontend/FYP_deployment after route parity is confirmed.

## App + Unified Merge Strategy

- Keep App.jsx as root shell and route switcher.
- Move UnifiedEmotionDetection.jsx to a page route, e.g. /emotion/live.
- Reuse CameraEmotionDetection and SpeechEmotionDetection inside UnifiedEmotionDetection as composable child blocks.
- Keep fallback error rendering and loading indicators from the updated components to preserve resilience with backend ML fallback responses.
