/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        clinical: {
          50: "#e8f5fe",
          100: "#b9e1fc",
          200: "#8bcdfa",
          300: "#5cb9f8",
          400: "#2ea5f6",
          500: "#0096FF",
          600: "#0078cc",
          700: "#005a99",
          800: "#003c66",
          900: "#001e33",
        },
      },
      animation: {
        "pulse-slow": "pulse 3s ease-in-out infinite",
        "scan-line": "scan-line 2s ease-in-out infinite",
      },
      keyframes: {
        "scan-line": {
          "0%, 100%": { transform: "translateY(-100%)" },
          "50%": { transform: "translateY(100%)" },
        },
      },
    },
  },
  plugins: [],
};
