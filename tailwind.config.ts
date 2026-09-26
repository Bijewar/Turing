import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
    "./public/**/*.html",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['var(--font-space-grotesk)', 'sans-serif'],
        mono: ['var(--font-jetbrains-mono)', 'monospace'],
      },
      colors: {
        black: '#000000',
        white: '#FFFFFF',
        orange: {
          500: '#FF4500', // Our primary orange
          600: '#E63E00', 
        },
        gray: {
          light: '#F5F5F5',
          text: '#666666',
        },
        border: '#E5E5E5',
        node: {
          controllable: '#FF6B35',
          mediator: '#004E89',
          bottleneck: '#C5283D',
          outcome: '#1A936F',
          chemistry: '#7B2D8E',
        },
        highlight: '#E91E63',
      },
      boxShadow: {
        'brutal-subtle': '4px 4px 0px rgba(0,0,0,0.05)',
        'brutal-medium': '4px 4px 0px #000000',
        'brutal-down': '0px 8px 0px rgba(0,0,0,0.1)',
        'brutal-warning': '8px 8px 0px #FF4500',
        'brutal-max': '16px 16px 0px #FF4500',
        'brutal-success': '4px 4px 0px #22C55E',
        'brutal-btn': '4px 4px 0px rgba(0,0,0,0.2)',
      },
      keyframes: {
        fadeIn: {
          'from': { opacity: '0', transform: 'translateY(5px)' },
          'to': { opacity: '1', transform: 'translateY(0)' },
        },
        flow: {
          'to': { strokeDashoffset: '-8' },
        },
        pulseSubtle: {
          '0%': { transform: 'scale(1)', opacity: '1', strokeWidth: '2.5px' },
          '50%': { transform: 'scale(1.15)', opacity: '0.9', strokeWidth: '4px', stroke: 'rgba(233, 30, 99, 0.5)' },
          '100%': { transform: 'scale(1)', opacity: '1', strokeWidth: '2.5px' },
        },
        fastScroll: {
          '0%': { transform: 'translateY(0)' },
          '100%': { transform: 'translateY(-50%)' },
        },
        slideInLock: {
          '0%': { transform: 'scale(3) translateY(100px)', opacity: '0' },
          '100%': { transform: 'scale(1) translateY(0)', opacity: '1' },
        },
        redactFlash: {
          '0%': { backgroundColor: 'transparent', color: '#000000' },
          '10%, 90%': { backgroundColor: '#000000', color: '#000000' },
          '100%': { backgroundColor: '#000000', color: '#FFFFFF' },
        },
        glitchAnim: {
          '0%': { transform: 'translate(0)' },
          '20%': { transform: 'translate(-2px, 2px)' },
          '40%': { transform: 'translate(-2px, -2px)' },
          '60%': { transform: 'translate(2px, 2px)' },
          '80%': { transform: 'translate(2px, -2px)' },
          '100%': { transform: 'translate(0)' },
        },
        typing: {
          'from': { width: '0' },
          'to': { width: '100%' },
        },
        blinkCaret: {
          'from, to': { borderColor: 'transparent' },
          '50%': { borderColor: '#FF4500' },
        },
        drawLine: {
          'to': { strokeDashoffset: '0' },
        },
        fadeNode: {
          'to': { opacity: '1', transform: 'translateY(0)' },
        },
        drawPath: {
          '0%': { strokeDashoffset: '1000' },
          '100%': { strokeDashoffset: '0' },
        },
        smoothPulse: {
          '0%, 100%': { opacity: '0.3' },
          '50%': { opacity: '0.8' },
        },
        subtleScan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(5px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease forwards',
        'flow': 'flow 0.8s linear infinite',
        'pulse-subtle': 'pulseSubtle 1.5s infinite ease-in-out',
        'fast-scroll': 'fastScroll 5s linear infinite',
        'slide-in-lock': 'slideInLock 0.4s cubic-bezier(0.1, 0.7, 0.1, 1) forwards',
        'redact': 'redactFlash 2s forwards',
        'glitch': 'glitchAnim 0.2s 5',
        'typewriter': 'typing 3s steps(40, end), blinkCaret 0.75s step-end infinite',
        'draw-line': 'drawLine 1.5s cubic-bezier(0.4, 0, 0.2, 1) forwards',
        'fade-node': 'fadeNode 0.8s ease-out forwards',
        'draw-path': 'drawPath 4s cubic-bezier(0.4, 0, 0.2, 1) infinite alternate',
        'smooth-pulse': 'smoothPulse 3s ease-in-out infinite',
        'subtle-scan': 'subtleScan 6s ease-in-out infinite alternate',
        'fade-up': 'fadeUp 0.5s ease-out forwards',
      },
    },
  },
  plugins: [],
};
export default config;
