// Static export: the whole site is plain files, so it can live on GitHub Pages,
// Vercel, Netlify or any CDN. Set NEXT_PUBLIC_BASE_PATH when serving from a
// sub-path such as https://<user>.github.io/roomwalk.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  trailingSlash: true,
  basePath,
  images: { unoptimized: true },
  reactStrictMode: true,
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
};

export default nextConfig;
