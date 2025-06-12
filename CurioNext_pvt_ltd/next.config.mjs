/** @type {import('next').NextConfig} */
const nextConfig = {
  // Ensure there are no redirects causing issues
  async redirects() {
    return []
  },
  // Make sure we're not using rewrites that could cause issues
  async rewrites() {
    return []
  },
  // Ensure trailing slashes are handled consistently
  trailingSlash: false,
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
