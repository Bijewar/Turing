/** @type {import('next').NextConfig} */
const API_BASE = process.env.BACKEND_API_URL ?? "http://127.0.0.1:8000";

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  compiler: {
    removeConsole: process.env.NODE_ENV === "production" ? { exclude: ["error"] } : false,
  },
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API_BASE}/api/:path*` },
      { source: "/health", destination: `${API_BASE}/health` },
    ];
  },
  images: {
    remotePatterns: [],
  },
  experimental: {
    serverActions: { bodySizeLimit: "100mb" },
  },
};

export default nextConfig;
