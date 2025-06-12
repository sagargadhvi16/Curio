"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Chart } from "@/components/ui/chart"
import { Bell, Brain, Clock, X } from "lucide-react"

export function AnalyticsSidebar({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [activeTab, setActiveTab] = useState("insights")

  // Sample data for mini charts
  const weeklyTrendData = {
    labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    datasets: [
      {
        label: "Questions Asked",
        data: [12, 19, 8, 15, 20, 14, 11],
        borderColor: "hsl(var(--primary))",
        backgroundColor: "hsla(var(--primary), 0.2)",
        fill: true,
        tension: 0.4,
      },
    ],
  }

  const timeOfDayData = {
    labels: ["Morning", "Afternoon", "Evening", "Night"],
    datasets: [
      {
        label: "Curiosity Level",
        data: [30, 45, 65, 20],
        backgroundColor: [
          "hsla(var(--primary), 0.6)",
          "hsla(var(--primary), 0.7)",
          "hsla(var(--primary), 0.8)",
          "hsla(var(--primary), 0.5)",
        ],
        borderColor: ["hsl(var(--primary))", "hsl(var(--primary))", "hsl(var(--primary))", "hsl(var(--primary))"],
        borderWidth: 1,
      },
    ],
  }

  // Sample notifications
  const notifications = [
    {
      id: 1,
      title: "New Interest Detected",
      description: "Avani is showing interest in astronomy",
      time: "10 minutes ago",
      type: "interest",
    },
    {
      id: 2,
      title: "Learning Milestone",
      description: "Avani has explored 5 new topics this week",
      time: "2 hours ago",
      type: "milestone",
    },
    {
      id: 3,
      title: "Weekly Summary Ready",
      description: "Listen to Avani's learning journey",
      time: "Yesterday",
      type: "summary",
    },
  ]

  // Sample quick insights
  const quickInsights = [
    {
      id: 1,
      title: "Peak Curiosity Time",
      description: "Avani is most curious in the evenings (6-8 PM)",
      icon: Clock,
    },
    {
      id: 2,
      title: "Growing Interest",
      description: "Interest in 'How Things Work' has increased by 10%",
      icon: Brain,
    },
    {
      id: 3,
      title: "Learning Streak",
      description: "Avani has been curious for 7 consecutive days",
      icon: Brain,
    },
  ]

  if (!isOpen) return null

  return (
    <div className="fixed inset-y-0 right-0 w-80 bg-background border-l border-border shadow-lg z-50 overflow-auto">
      <div className="p-4 border-b sticky top-0 bg-background z-10 flex justify-between items-center">
        <h2 className="font-semibold">Quick Analytics</h2>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      <div className="p-4">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="w-full mb-4">
            <TabsTrigger value="insights" className="flex-1">
              Insights
            </TabsTrigger>
            <TabsTrigger value="notifications" className="flex-1">
              Notifications
              <Badge className="ml-2 h-5 w-5 p-0 flex items-center justify-center">{notifications.length}</Badge>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="insights" className="space-y-4">
            {quickInsights.map((insight) => (
              <Card key={insight.id}>
                <CardContent className="p-3">
                  <div className="flex items-start gap-3">
                    <div className="bg-primary/10 p-2 rounded-full">
                      <insight.icon className="h-4 w-4 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-medium text-sm">{insight.title}</h3>
                      <p className="text-xs text-muted-foreground">{insight.description}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}

            <div className="space-y-1">
              <h3 className="text-sm font-medium">Weekly Trend</h3>
              <div className="h-32">
                <Chart
                  type="line"
                  data={weeklyTrendData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        display: false,
                      },
                    },
                    scales: {
                      x: {
                        grid: {
                          display: false,
                        },
                      },
                      y: {
                        beginAtZero: true,
                        grid: {
                          display: false,
                        },
                      },
                    },
                  }}
                />
              </div>
            </div>

            <div className="space-y-1">
              <h3 className="text-sm font-medium">Time of Day Activity</h3>
              <div className="h-32">
                <Chart
                  type="doughnut"
                  data={timeOfDayData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: "right",
                        labels: {
                          boxWidth: 10,
                          font: {
                            size: 10,
                          },
                        },
                      },
                    },
                  }}
                />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="notifications">
            <div className="space-y-3">
              {notifications.map((notification) => (
                <Card key={notification.id}>
                  <CardContent className="p-3">
                    <div className="flex items-start gap-3">
                      <div className="bg-primary/10 p-2 rounded-full">
                        <Bell className="h-4 w-4 text-primary" />
                      </div>
                      <div>
                        <h3 className="font-medium text-sm">{notification.title}</h3>
                        <p className="text-xs text-muted-foreground">{notification.description}</p>
                        <p className="text-xs text-muted-foreground mt-1">{notification.time}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
