"use client"

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { RefreshCw } from "lucide-react"
import { useEffect, useState } from "react"

type props = { onRefresh: () => void }
export function DashboardHeader({ onRefresh }: props) {
  const [greeting, setGreeting] = useState("Good morning")
  const [date, setDate] = useState("")
  const [isRefreshing, setIsRefreshing] = useState(false)

  useEffect(() => {
    const now = new Date()
    const hour = now.getHours()

    if (hour < 12) {
      setGreeting("Good morning")
    } else if (hour < 18) {
      setGreeting("Good afternoon")
    } else {
      setGreeting("Good evening")
    }

    setDate(
      now.toLocaleDateString("en-US", {
        weekday: "long",
        month: "long",
        day: "numeric",
      }),
    )
  }, [])

  const handleRefresh = () => {
    onRefresh()
    setIsRefreshing(true)
    // Simulate refresh
    setTimeout(() => {
      setIsRefreshing(false)
    }, 1000)
  }

  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div className="flex items-center gap-4">
            <Avatar className="h-16 w-16">
              <AvatarImage src="/placeholder.svg?height=64&width=64" alt="Child's avatar" />
              <AvatarFallback>CN</AvatarFallback>
            </Avatar>
            <div>
              <h1 className="text-2xl font-bold">{greeting}, Swati</h1>
              <p className="text-muted-foreground">{date}</p>
              <p className="text-sm mt-1">
                Avani has explored 3 new topics this week, including a new interest in painting
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" onClick={handleRefresh} disabled={isRefreshing}>
              <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
