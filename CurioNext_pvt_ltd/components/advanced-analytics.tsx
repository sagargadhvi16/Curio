"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Chart } from "@/components/ui/chart"
import { Button } from "@/components/ui/button"
import { Download, Clock } from "lucide-react"

// Sample data for heatmap
const heatmapData = {
  labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
  datasets: [
    {
      label: "Morning",
      data: [30, 45, 25, 35, 40, 60, 50],
      backgroundColor: "rgba(255, 99, 132, 0.8)",
    },
    {
      label: "Afternoon",
      data: [50, 35, 40, 45, 35, 40, 30],
      backgroundColor: "rgba(54, 162, 235, 0.8)",
    },
    {
      label: "Evening",
      data: [70, 60, 55, 65, 75, 50, 40],
      backgroundColor: "rgba(75, 192, 192, 0.8)",
    },
    {
      label: "Night",
      data: [20, 25, 30, 15, 25, 30, 20],
      backgroundColor: "rgba(153, 102, 255, 0.8)",
    },
  ],
}

// Sample data for category engagement
const categoryEngagementData = {
  labels: ["How Things Work", "Big Questions", "Creativity", "World & Cultures", "Nature & Life", "Self & Emotions"],
  datasets: [
    {
      label: "Last Month",
      data: [65, 40, 55, 35, 70, 30],
      backgroundColor: "rgba(255, 99, 132, 0.5)",
      borderColor: "rgba(255, 99, 132, 1)",
      borderWidth: 1,
    },
    {
      label: "This Month",
      data: [75, 35, 60, 30, 65, 40],
      backgroundColor: "rgba(54, 162, 235, 0.5)",
      borderColor: "rgba(54, 162, 235, 1)",
      borderWidth: 1,
    },
  ],
}

// Sample data for word cloud
const wordCloudData = [
  { text: "Dinosaurs", value: 100 },
  { text: "Space", value: 80 },
  { text: "Volcanoes", value: 75 },
  { text: "Animals", value: 70 },
  { text: "Planets", value: 65 },
  { text: "Robots", value: 60 },
  { text: "Oceans", value: 55 },
  { text: "Stars", value: 50 },
  { text: "Insects", value: 45 },
  { text: "Weather", value: 40 },
  { text: "Plants", value: 35 },
  { text: "Electricity", value: 30 },
  { text: "Magnets", value: 25 },
  { text: "Fossils", value: 20 },
  { text: "Rockets", value: 15 },
]

export function AdvancedAnalytics() {
  const [timeRange, setTimeRange] = useState("month")
  const [activeTab, setActiveTab] = useState("heatmap")

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Advanced Analytics</CardTitle>
          <CardDescription>Deeper insights into your child's curiosity patterns</CardDescription>
        </div>
        <div className="flex items-center gap-2">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-[120px]">
              <SelectValue placeholder="Time range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="week">This Week</SelectItem>
              <SelectItem value="month">This Month</SelectItem>
              <SelectItem value="quarter">This Quarter</SelectItem>
              <SelectItem value="year">This Year</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" size="icon">
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-4">
            <TabsTrigger value="heatmap">Time Heatmap</TabsTrigger>
            <TabsTrigger value="category">Category Engagement</TabsTrigger>
            <TabsTrigger value="wordcloud">Common Themes</TabsTrigger>
          </TabsList>

          <TabsContent value="heatmap">
            <div className="mb-2 text-sm text-muted-foreground">
              When your child is most curious throughout the week
            </div>
            <div className="h-[350px] relative">
              <Chart
                type="bar"
                data={heatmapData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    x: {
                      title: {
                        display: true,
                        text: "Day of Week",
                      },
                    },
                    y: {
                      beginAtZero: true,
                      title: {
                        display: true,
                        text: "Curiosity Level",
                      },
                    },
                  },
                  plugins: {
                    title: {
                      display: true,
                      text: "Curiosity Heatmap by Time of Day",
                    },
                    tooltip: {
                      callbacks: {
                        label: (context) => {
                          const label = context.dataset.label || ""
                          const value = context.raw || 0
                          return `${label}: ${value}% curiosity level`
                        },
                      },
                    },
                  },
                }}
              />
            </div>
            <div className="mt-6 text-sm">
              <div className="flex items-start gap-2">
                <Clock className="h-4 w-4 text-primary mt-0.5" />
                <div>
                  <p className="font-medium">Time Investment:</p>
                  <p className="text-muted-foreground mt-1">
                    Avani spends the most time exploring dinosaurs (4.5 hours/week), followed by space (3.2 hours/week).
                  </p>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="category">
            <div className="mb-2 text-sm text-muted-foreground">Comparing category engagement between time periods</div>
            <div className="h-[350px] relative">
              <Chart
                type="bar"
                data={categoryEngagementData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    x: {
                      title: {
                        display: true,
                        text: "Interest Categories",
                      },
                    },
                    y: {
                      beginAtZero: true,
                      title: {
                        display: true,
                        text: "Engagement Level (%)",
                      },
                    },
                  },
                  plugins: {
                    title: {
                      display: true,
                      text: "Category Engagement Comparison",
                    },
                  },
                }}
              />
            </div>
            <div className="mt-6 text-sm">
              <p className="font-medium">Key Insight:</p>
              <p className="text-muted-foreground">
                Avani's interest in "How Things Work" has increased by 10% compared to last month.
              </p>
            </div>
          </TabsContent>

          <TabsContent value="wordcloud">
            <div className="mb-2 text-sm text-muted-foreground">Most common themes in your child's questions</div>
            <div className="h-[350px] bg-muted/30 rounded-md flex items-center justify-center p-4 relative">
              <div className="relative w-full h-full">
                {wordCloudData.map((word, index) => {
                  // Calculate position and size based on value
                  const size = Math.max(14, Math.min(36, word.value / 3))
                  const opacity = Math.max(0.5, word.value / 100)

                  // Generate pseudo-random positions
                  const seed = index * 137.5
                  const x = (Math.sin(seed) * 0.4 + 0.5) * 100
                  const y = (Math.cos(seed) * 0.4 + 0.5) * 100

                  return (
                    <div
                      key={word.text}
                      className="absolute transform -translate-x-1/2 -translate-y-1/2 font-medium"
                      style={{
                        left: `${x}%`,
                        top: `${y}%`,
                        fontSize: `${size}px`,
                        opacity: opacity,
                        color: `hsl(var(--primary) / ${opacity})`,
                        zIndex: Math.floor(word.value / 10),
                      }}
                    >
                      {word.text}
                    </div>
                  )
                })}
              </div>
            </div>
            <div className="mt-6 text-sm">
              <p className="font-medium">Key Insight:</p>
              <p className="text-muted-foreground">
                Dinosaurs, space, and volcanoes are the most common themes in Avani's questions.
              </p>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
