"use client"

import type React from "react"
import { ModeToggle } from "@/components/mode-toggle"
import { Logo } from "@/components/logo"
import { AiAssistant } from "@/components/ai-assistant"

export function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-10 flex h-16 items-center gap-4 border-b bg-background px-6">
        <div className="flex items-center gap-2">
          <Logo size="sm" />
          <div>
            <h1 className="font-bold text-lg bg-gradient-to-r from-amber-500 to-amber-700 dark:from-amber-400 dark:to-amber-600 bg-clip-text text-transparent">
              CurioNext
            </h1>
            <p className="text-xs text-muted-foreground">Nurturing Childhood Curiosity</p>
          </div>
        </div>
        <div className="ml-auto">
          <ModeToggle />
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1">{children}</main>

      {/* AI Assistant */}
      <AiAssistant />
    </div>
  )
}
