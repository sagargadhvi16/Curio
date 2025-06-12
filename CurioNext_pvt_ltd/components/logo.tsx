"use client"

import Image from "next/image"
import Link from "next/link"

interface LogoProps {
  size?: "sm" | "md" | "lg" | "xl"
  withText?: boolean
  className?: string
  onClick?: () => void
}

export function Logo({ size = "md", withText = true, className = "", onClick }: LogoProps) {
  const sizes = {
    sm: { container: "h-10 w-10", text: "text-base" },
    md: { container: "h-12 w-12", text: "text-lg" },
    lg: { container: "h-16 w-16", text: "text-2xl" },
    xl: { container: "h-20 w-20", text: "text-3xl" },
  }

  return (
    <Link href="/" className={`flex items-center gap-3 ${className}`} onClick={onClick}>
      <div
        className={`relative flex ${sizes[size].container} items-center justify-center overflow-hidden rounded-xl bg-gradient-to-br from-amber-400 to-amber-500 shadow-md border-2 border-amber-300 dark:border-amber-600`}
      >
        <Image
          src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/curionext-high-resolution-logo-UlyGnnH7xKKOIsZUrsuoO2gVadu0rT.png"
          alt="CurioNext Logo"
          width={size === "xl" ? 80 : size === "lg" ? 64 : size === "md" ? 48 : 40}
          height={size === "xl" ? 80 : size === "lg" ? 64 : size === "md" ? 48 : 40}
          className="object-contain scale-125"
          priority
        />
      </div>
      {withText && (
        <div className="flex flex-col">
          <span
            className={`font-bold ${sizes[size].text} bg-gradient-to-r from-amber-500 to-amber-700 dark:from-amber-400 dark:to-amber-600 bg-clip-text text-transparent`}
          >
            
          </span>
          {size === "lg" || size === "xl" ? (
            <span className="text-xs text-muted-foreground">Nurturing Childhood Curiosity</span>
          ) : null}
        </div>
      )}
    </Link>
  )
}
