/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{vue,js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                background: '#000000', // Strict Black
                surface: '#111111',    // Near Black
                border: '#333333',     // Dark Grey
                primary: '#ff5722',    // Deep Orange
                secondary: '#ffffff',  // White Text
                muted: '#666666',      // Muted Text
                success: '#ff9100',    // Orange-Yellow (No Green)
                error: '#ff3d00',      // Red-Orange
                warning: '#ffc107',    // Amber
                info: '#2979ff',       // Blue (Keep slightly for info, but desaturated)
            },
            fontFamily: {
                sans: ['"Titillium Web"', 'sans-serif'],
                mono: ['"JetBrains Mono"', 'monospace'],
            },
        },
    },
    plugins: [],
}
