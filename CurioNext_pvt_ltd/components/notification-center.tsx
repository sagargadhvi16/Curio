"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import {
  Bell,
  Search,
  Filter,
  CheckCircle2,
  Clock,
  Trash2,
  Brain,
  Award,
  Calendar,
  AlertTriangle,
  BookOpen,
  Settings,
  ChevronRight,
} from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"

// Sample notification data
const initialNotifications = [
  {
    id: 1,
    title: "New Interest Detected",
    description: "Avani has been asking questions about astronomy and space exploration.",
    time: "10 minutes ago",
    date: "2025-03-22T10:30:00",
    type: "interest",
    read: false,
    icon: Brain,
    category: "learning",
  },
  {
    id: 2,
    title: "Learning Milestone Achieved",
    description: "Avani has explored 10 different topics in 'How Things Work' category.",
    time: "2 hours ago",
    date: "2025-03-22T09:15:00",
    type: "milestone",
    read: false,
    icon: Award,
    category: "learning",
  },
  {
    id: 3,
    title: "Weekly Summary Available",
    description: "Your child's weekly learning summary for March 15-21 is now available.",
    time: "Yesterday",
    date: "2025-03-21T14:20:00",
    type: "summary",
    read: true,
    icon: Calendar,
    category: "report",
  },
  {
    id: 4,
    title: "Safety Alert",
    description: "Avani attempted to access restricted content about violent video games.",
    time: "2 days ago",
    date: "2025-03-20T11:45:00",
    type: "alert",
    read: true,
    icon: AlertTriangle,
    category: "safety",
  },
  {
    id: 5,
    title: "New Suggested Activity",
    description: "Based on Avani's interest in dinosaurs, we've added a new activity.",
    time: "3 days ago",
    date: "2025-03-19T16:30:00",
    type: "activity",
    read: true,
    icon: BookOpen,
    category: "learning",
  },
  {
    id: 6,
    title: "Privacy Settings Updated",
    description: "The system privacy policy has been updated. Please review the changes.",
    time: "1 week ago",
    date: "2025-03-15T09:00:00",
    type: "system",
    read: true,
    icon: Settings,
    category: "system",
  },
  {
    id: 7,
    title: "New Interest Trend",
    description: "Avani's interest in marine biology has increased by 30% this month.",
    time: "1 week ago",
    date: "2025-03-15T10:45:00",
    type: "interest",
    read: true,
    icon: Brain,
    category: "learning",
  },
  {
    id: 8,
    title: "Question Pattern Detected",
    description: "Avani has been asking more 'why' questions, showing deeper curiosity.",
    time: "2 weeks ago",
    date: "2025-03-08T13:20:00",
    type: "insight",
    read: true,
    icon: Brain,
    category: "learning",
  },
]

export function NotificationCenter() {
  const [notifications, setNotifications] = useState(initialNotifications)
  const [activeTab, setActiveTab] = useState("all")
  const [searchQuery, setSearchQuery] = useState("")
  const [showFilters, setShowFilters] = useState(false)
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [selectedTimeframe, setSelectedTimeframe] = useState<string | null>(null)
  const [showUnreadOnly, setShowUnreadOnly] = useState(false)

  const unreadCount = notifications.filter((n) => !n.read).length

  const handleMarkAsRead = (id: number) => {
    setNotifications(
      notifications.map((notification) => (notification.id === id ? { ...notification, read: true } : notification)),
    )
  }

  const handleMarkAllAsRead = () => {
    setNotifications(notifications.map((notification) => ({ ...notification, read: true })))
  }

  const handleDeleteNotification = (id: number) => {
    setNotifications(notifications.filter((notification) => notification.id !== id))
  }

  const handleClearAll = () => {
    setNotifications([])
  }

  const resetFilters = () => {
    setSearchQuery("")
    setSelectedCategory(null)
    setSelectedTimeframe(null)
    setShowUnreadOnly(false)
  }

  const filteredNotifications = notifications.filter((notification) => {
    // Filter by search query
    if (
      searchQuery &&
      !notification.title.toLowerCase().includes(searchQuery.toLowerCase()) &&
      !notification.description.toLowerCase().includes(searchQuery.toLowerCase())
    ) {
      return false
    }

    // Filter by tab
    if (activeTab !== "all" && notification.category !== activeTab) {
      return false
    }

    // Filter by category
    if (selectedCategory && notification.type !== selectedCategory) {
      return false
    }

    // Filter by timeframe
    if (selectedTimeframe) {
      const notificationDate = new Date(notification.date)
      const now = new Date()

      if (selectedTimeframe === "today") {
        const today = new Date()
        today.setHours(0, 0, 0, 0)
        if (notificationDate < today) return false
      } else if (selectedTimeframe === "week") {
        const weekAgo = new Date()
        weekAgo.setDate(weekAgo.getDate() - 7)
        if (notificationDate < weekAgo) return false
      } else if (selectedTimeframe === "month") {
        const monthAgo = new Date()
        monthAgo.setMonth(monthAgo.getMonth() - 1)
        if (notificationDate < monthAgo) return false
      }
    }

    // Filter by read status
    if (showUnreadOnly && notification.read) {
      return false
    }

    return true
  })

  const getNotificationBadge = (type: string) => {
    switch (type) {
      case "interest":
        return <Badge className="bg-blue-500">Interest</Badge>
      case "milestone":
        return <Badge className="bg-green-500">Milestone</Badge>
      case "summary":
        return <Badge variant="secondary">Summary</Badge>
      case "alert":
        return <Badge variant="destructive">Alert</Badge>
      case "activity":
        return <Badge className="bg-purple-500">Activity</Badge>
      case "system":
        return <Badge variant="outline">System</Badge>
      case "insight":
        return <Badge className="bg-amber-500">Insight</Badge>
      default:
        return <Badge variant="outline">{type}</Badge>
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <CardTitle>Notification Center</CardTitle>
            <CardDescription>Stay updated on your child's learning journey and important alerts</CardDescription>
          </div>
          <div className="flex flex-col sm:flex-row gap-2">
            {unreadCount > 0 && (
              <Button variant="outline" onClick={handleMarkAllAsRead}>
                <CheckCircle2 className="h-4 w-4 mr-2" />
                Mark All as Read
              </Button>
            )}
            <Button variant="outline" onClick={() => setShowFilters(!showFilters)}>
              <Filter className="h-4 w-4 mr-2" />
              Filters
            </Button>
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline" className="text-destructive">
                  <Trash2 className="h-4 w-4 mr-2" />
                  Clear All
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Clear All Notifications</DialogTitle>
                  <DialogDescription>
                    Are you sure you want to delete all notifications? This action cannot be undone.
                  </DialogDescription>
                </DialogHeader>
                <DialogFooter>
                  <Button variant="outline">Cancel</Button>
                  <Button variant="destructive" onClick={handleClearAll}>
                    Delete All
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
        </div>

        {showFilters && (
          <div className="mt-4 space-y-4 p-4 bg-muted/30 rounded-lg">
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search notifications..."
                  className="pl-8"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
              <Select value={selectedCategory || ""} onValueChange={(value) => setSelectedCategory(value || null)}>
                <SelectTrigger className="w-full sm:w-[180px]">
                  <SelectValue placeholder="Notification type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All types</SelectItem>
                  <SelectItem value="interest">Interest</SelectItem>
                  <SelectItem value="milestone">Milestone</SelectItem>
                  <SelectItem value="summary">Summary</SelectItem>
                  <SelectItem value="alert">Alert</SelectItem>
                  <SelectItem value="activity">Activity</SelectItem>
                  <SelectItem value="system">System</SelectItem>
                  <SelectItem value="insight">Insight</SelectItem>
                </SelectContent>
              </Select>
              <Select value={selectedTimeframe || ""} onValueChange={(value) => setSelectedTimeframe(value || null)}>
                <SelectTrigger className="w-full sm:w-[180px]">
                  <SelectValue placeholder="Time period" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All time</SelectItem>
                  <SelectItem value="today">Today</SelectItem>
                  <SelectItem value="week">Past week</SelectItem>
                  <SelectItem value="month">Past month</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Switch id="unread-only" checked={showUnreadOnly} onCheckedChange={setShowUnreadOnly} />
                <Label htmlFor="unread-only">Show unread only</Label>
              </div>
              <Button variant="ghost" size="sm" onClick={resetFilters}>
                Reset filters
              </Button>
            </div>
          </div>
        )}
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="all" value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-4">
            <TabsTrigger value="all" className="relative">
              All
              {unreadCount > 0 && (
                <Badge className="ml-2 h-5 w-5 p-0 flex items-center justify-center">{unreadCount}</Badge>
              )}
            </TabsTrigger>
            <TabsTrigger value="learning">Learning</TabsTrigger>
            <TabsTrigger value="safety">Safety</TabsTrigger>
            <TabsTrigger value="report">Reports</TabsTrigger>
            <TabsTrigger value="system">System</TabsTrigger>
          </TabsList>

          <TabsContent value={activeTab}>
            {filteredNotifications.length === 0 ? (
              <div className="text-center py-12">
                <Bell className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-medium mb-2">No notifications</h3>
                <p className="text-muted-foreground">
                  {searchQuery || selectedCategory || selectedTimeframe || showUnreadOnly
                    ? "No notifications match your current filters"
                    : "You're all caught up!"}
                </p>
                {(searchQuery || selectedCategory || selectedTimeframe || showUnreadOnly) && (
                  <Button variant="outline" className="mt-4" onClick={resetFilters}>
                    Clear Filters
                  </Button>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                {filteredNotifications.map((notification) => (
                  <div
                    key={notification.id}
                    className={`border rounded-lg p-4 transition-colors ${
                      notification.read ? "bg-background" : "bg-primary/5"
                    }`}
                  >
                    <div className="flex items-start gap-4">
                      <div className={`p-2 rounded-full ${notification.read ? "bg-muted" : "bg-primary/10"}`}>
                        <notification.icon
                          className={`h-5 w-5 ${notification.read ? "text-muted-foreground" : "text-primary"}`}
                        />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <h3 className="font-medium">{notification.title}</h3>
                            {getNotificationBadge(notification.type)}
                            {!notification.read && <span className="h-2 w-2 rounded-full bg-primary" />}
                          </div>
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Clock className="h-3 w-3" />
                            <span>{notification.time}</span>
                          </div>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">{notification.description}</p>
                      </div>
                      <div className="flex flex-col gap-2">
                        {!notification.read && (
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => handleMarkAsRead(notification.id)}
                            title="Mark as read"
                          >
                            <CheckCircle2 className="h-4 w-4" />
                          </Button>
                        )}
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => handleDeleteNotification(notification.id)}
                          title="Delete notification"
                        >
                          <Trash2 className="h-4 w-4 text-destructive" />
                        </Button>
                        <Button variant="ghost" size="icon" title="View details">
                          <ChevronRight className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
