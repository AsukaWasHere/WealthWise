module.exports = {
  content: [
    "./pages/*.{html,js}",
    "./index.html",
    "./src/**/*.{html,js,jsx,ts,tsx}",
    "./components/**/*.{html,js,jsx,ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        // Primary Colors - Deep trust foundation
        primary: {
          DEFAULT: "#1B365D", // primary-900
          50: "#F0F4F8", // slate-50
          100: "#D9E2EC", // slate-100
          200: "#BCCCDC", // slate-200
          300: "#9FB3C8", // slate-300
          400: "#829AB1", // slate-400
          500: "#627D98", // slate-500
          600: "#486581", // slate-600
          700: "#334E68", // slate-700
          800: "#243B53", // slate-800
          900: "#1B365D", // slate-900
        },
        
        // Secondary Colors - Supporting intelligence
        secondary: {
          DEFAULT: "#2E5984", // blue-700
          50: "#F1F5F9", // slate-50
          100: "#E2E8F0", // slate-100
          200: "#CBD5E0", // slate-200
          300: "#A0AEC0", // slate-300
          400: "#718096", // slate-400
          500: "#4A5568", // slate-500
          600: "#2E5984", // blue-700
          700: "#2C5282", // blue-800
          800: "#2A4365", // blue-900
          900: "#1A365D", // blue-900
        },

        // Accent Colors - Growth moments
        accent: {
          DEFAULT: "#00B4A6", // teal-600
          50: "#F0FDFA", // teal-50
          100: "#CCFBF1", // teal-100
          200: "#99F6E4", // teal-200
          300: "#5EEAD4", // teal-300
          400: "#2DD4BF", // teal-400
          500: "#14B8A6", // teal-500
          600: "#00B4A6", // teal-600
          700: "#0F766E", // teal-700
          800: "#115E59", // teal-800
          900: "#134E4A", // teal-900
        },

        // Background Colors
        background: "#FAFBFC", // gray-50
        surface: {
          DEFAULT: "#F5F7FA", // gray-100
          hover: "#EDF2F7", // gray-200
        },

        // Text Colors
        text: {
          primary: "#1A202C", // gray-900
          secondary: "#4A5568", // gray-600
          tertiary: "#718096", // gray-500
          inverse: "#FFFFFF", // white
        },

        // Status Colors
        success: {
          DEFAULT: "#38A169", // green-600
          50: "#F0FFF4", // green-50
          100: "#C6F6D5", // green-100
          200: "#9AE6B4", // green-200
          300: "#68D391", // green-300
          400: "#48BB78", // green-400
          500: "#38A169", // green-500
          600: "#2F855A", // green-600
          700: "#276749", // green-700
          800: "#22543D", // green-800
        },

        warning: {
          DEFAULT: "#D69E2E", // yellow-600
          50: "#FFFBEB", // yellow-50
          100: "#FEF5E7", // yellow-100
          200: "#FED7AA", // yellow-200
          300: "#FDBA74", // yellow-300
          400: "#F59E0B", // yellow-400
          500: "#D69E2E", // yellow-500
          600: "#B45309", // yellow-600
          700: "#92400E", // yellow-700
          800: "#78350F", // yellow-800
        },

        error: {
          DEFAULT: "#E53E3E", // red-500
          50: "#FED7D7", // red-100
          100: "#FEB2B2", // red-200
          200: "#FC8181", // red-300
          300: "#F56565", // red-400
          400: "#E53E3E", // red-500
          500: "#C53030", // red-600
          600: "#9B2C2C", // red-700
          700: "#742A2A", // red-800
        },

        // Border Colors
        border: {
          DEFAULT: "#E2E8F0", // gray-200
          light: "#F7FAFC", // gray-100
          dark: "#CBD5E0", // gray-300
        },
      },

      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        inter: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Courier New', 'monospace'],
        jetbrains: ['JetBrains Mono', 'Courier New', 'monospace'],
      },

      fontSize: {
        'xs': ['0.75rem', { lineHeight: '1rem' }],
        'sm': ['0.875rem', { lineHeight: '1.25rem' }],
        'base': ['1rem', { lineHeight: '1.5rem' }],
        'lg': ['1.125rem', { lineHeight: '1.75rem' }],
        'xl': ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
        '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
        '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
        '5xl': ['3rem', { lineHeight: '1.2' }],
        '6xl': ['3.75rem', { lineHeight: '1.2' }],
        '7xl': ['4.5rem', { lineHeight: '1.1' }],
        '8xl': ['6rem', { lineHeight: '1.1' }],
        '9xl': ['8rem', { lineHeight: '1.1' }],
      },

      fontWeight: {
        normal: '400',
        medium: '500',
        semibold: '600',
        bold: '700',
      },

      boxShadow: {
        'sm': '0 1px 2px 0 rgba(27, 54, 93, 0.05)',
        'DEFAULT': '0 1px 3px 0 rgba(27, 54, 93, 0.08), 0 1px 2px 0 rgba(27, 54, 93, 0.06)',
        'md': '0 4px 6px -1px rgba(27, 54, 93, 0.08), 0 2px 4px -1px rgba(27, 54, 93, 0.06)',
        'lg': '0 10px 15px -3px rgba(27, 54, 93, 0.08), 0 4px 6px -2px rgba(27, 54, 93, 0.05)',
        'xl': '0 20px 25px -5px rgba(27, 54, 93, 0.08), 0 10px 10px -5px rgba(27, 54, 93, 0.04)',
        '2xl': '0 25px 50px -12px rgba(27, 54, 93, 0.12)',
        'floating': '0 4px 20px rgba(27, 54, 93, 0.08)',
        'none': 'none',
      },

      borderRadius: {
        'none': '0',
        'sm': '0.125rem',
        'DEFAULT': '0.25rem',
        'md': '0.375rem',
        'lg': '0.5rem',
        'xl': '0.75rem',
        '2xl': '1rem',
        '3xl': '1.5rem',
        'full': '9999px',
      },

      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },

      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'scale-in': 'scaleIn 0.3s ease-out',
      },

      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
      },

      transitionDuration: {
        '150': '150ms',
        '300': '300ms',
        '500': '500ms',
      },

      transitionTimingFunction: {
        'ease-out': 'cubic-bezier(0, 0, 0.2, 1)',
      },

      backdropBlur: {
        xs: '2px',
      },

      zIndex: {
        '60': '60',
        '70': '70',
        '80': '80',
        '90': '90',
        '100': '100',
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}