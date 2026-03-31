/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        medical: {
          blue: { 
            light: '#E3F2FD', 
            DEFAULT: '#2196F3', 
            dark: '#1976D2', 
            darker: '#0D47A1' 
          },
          white: '#FFFFFF',
          gray: { 
            light: '#F5F5F5', 
            DEFAULT: '#9E9E9E', 
            dark: '#616161' 
          }
        }
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        }
      }
    },
  },
  plugins: [],
}
