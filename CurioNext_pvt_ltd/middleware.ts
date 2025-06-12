import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

export function middleware(request: NextRequest) {
  // Get the pathname of the request
  const pathname = request.nextUrl.pathname

  // If the pathname is just '/' with no trailing slash, we don't need to redirect
  if (pathname === "/") {
    return NextResponse.next()
  }

  // Remove any trailing slashes for consistent routing
  if (pathname.endsWith("/") && pathname.length > 1) {
    const url = new URL(request.url)
    url.pathname = pathname.slice(0, -1)
    return NextResponse.redirect(url)
  }

  return NextResponse.next()
}

// See: https://nextjs.org/docs/app/building-your-application/routing/middleware#matcher
export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    "/((?!api|_next/static|_next/image|favicon.ico).*)",
  ],
}
