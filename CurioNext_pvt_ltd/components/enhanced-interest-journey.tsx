"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Chart } from "@/components/ui/chart"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Clock, TrendingUp, ArrowRight, Calendar } from "lucide-react"

// Sample data for interest frequency over time
const frequencyData = {
  labels: ["Week 1", "Week 2", "Week 3", "Week 4"],
  datasets: [
    {
      label: "Dinosaurs",
      data: [15, 20, 25, 30],
      borderColor: "rgb(255, 107, 107)",
      backgroundColor: "rgba(255, 107, 107, 0.1)",
      fill: true,
    },
    {
      label: "Space",
      data: [10, 15, 20, 25],
      borderColor: "rgb(78, 205, 196)",
      backgroundColor: "rgba(78, 205, 196, 0.1)",
      fill: true,
    },
    {
      label: "Volcanoes",
      data: [5, 10, 15, 20],
      borderColor: "rgb(255, 209, 102)",
      backgroundColor: "rgba(255, 209, 102, 0.1)",
      fill: true,
    },
  ],
}

// Sample data for interest correlation
const correlationData = [
  { interest1: "Dinosaurs", interest2: "Fossils", correlation: 0.85 },
  { interest1: "Dinosaurs", interest2: "Ancient Animals", correlation: 0.75 },
  { interest1: "Space", interest2: "Planets", correlation: 0.9 },
  { interest1: "Space", interest2: "Stars", correlation: 0.8 },
  { interest1: "Volcanoes", interest2: "Earthquakes", correlation: 0.7 },
  { interest1: "Volcanoes", interest2: "Mountains", correlation: 0.65 },
]

// Sample data for time spent
const timeSpentData = {
  labels: ["Dinosaurs", "Space", "Volcanoes", "Animals", "Oceans", "Weather"],
  datasets: [
    {
      label: "Hours per Week",
      data: [4.5, 3.2, 2.8, 2.0, 1.5, 1.0],
      backgroundColor: [
        "rgba(255, 107, 107, 0.7)",
        "rgba(78, 205, 196, 0.7)",
        "rgba(255, 209, 102, 0.7)",
        "rgba(106, 5, 114, 0.7)",
        "rgba(26, 147, 111, 0.7)",
        "rgba(192, 108, 132, 0.7)",
      ],
      borderColor: [
        "rgb(255, 107, 107)",
        "rgb(78, 205, 196)",
        "rgb(255, 209, 102)",
        "rgb(106, 5, 114)",
        "rgb(26, 147, 111)",
        "rgb(192, 108, 132)",
      ],
      borderWidth: 1,
    },
  ],
}

// Sample data for interest evolution
const evolutionData = {
  labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
  datasets: [
    {
      label: "Dinosaurs",
      data: [30, 40, 45, 50, 45, 40],
      borderColor: "rgb(255, 107, 107)",
      backgroundColor: "rgba(255, 107, 107, 0.1)",
      fill: false,
    },
    {
      label: "Space",
      data: [10, 15, 25, 35, 45, 55],
      borderColor: "rgb(78, 205, 196)",
      backgroundColor: "rgba(78, 205, 196, 0.1)",
      fill: false,
    },
    {
      label: "Volcanoes",
      data: [5, 15, 30, 35, 30, 25],
      borderColor: "rgb(255, 209, 102)",
      backgroundColor: "rgba(255, 209, 102, 0.1)",
      fill: false,
    },
    {
      label: "Oceans",
      data: [0, 5, 10, 15, 25, 35],
      borderColor: "rgb(26, 147, 111)",
      backgroundColor: "rgba(26, 147, 111, 0.1)",
      fill: false,
    },
  ],
}

export function EnhancedInterestJourney() {
  const [timeRange, setTimeRange] = useState("month")

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Enhanced Interest Journey</CardTitle>
          <CardDescription>
            Detailed analysis of your child's evolving interests and engagement patterns
          </CardDescription>
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
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="frequency">
          <TabsList className="mb-4">
            <TabsTrigger value="frequency">Question Frequency</TabsTrigger>
            <TabsTrigger value="correlation">Interest Correlation</TabsTrigger>
            <TabsTrigger value="timespent">Time Spent</TabsTrigger>
            <TabsTrigger value="evolution">Interest Evolution</TabsTrigger>
          </TabsList>

          <TabsContent value="frequency" className="h-[350px] relative">
            <Chart
              type="line"
              data={frequencyData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  x: {
                    title: {
                      display: true,
                      text: "Time Period",
                    },
                  },
                  y: {
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: "Number of Questions",
                    },
                  },
                },
                plugins: {
                  tooltip: {
                    mode: "index",
                    intersect: false,
                  },
                },
              }}
            />
            <div className="mt-4 text-sm">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-primary" />
                <p className="font-medium">Key Insight:</p>
              </div>
              <p className="text-muted-foreground ml-6">
                Questions about dinosaurs have consistently increased over the past month, showing a deepening interest.
              </p>
            </div>
          </TabsContent>

          <TabsContent value="correlation">
            <div className="h-[350px] overflow-auto">
              <div className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  This chart shows how interests are connected. A higher correlation means interests are often explored
                  together.
                </p>
                {correlationData.map((item, index) => (
                  <div key={index} className="bg-muted/30 p-3 rounded-md">
                    <div className="flex justify-between items-center mb-2">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{item.interest1}</span>
                        <ArrowRight className="h-4 w-4 text-muted-foreground" />
                        <span className="font-medium">{item.interest2}</span>
                      </div>
                      <Badge variant={item.correlation > 0.7 ? "default" : "outline"}>
                        {Math.round(item.correlation * 100)}% correlation
                      </Badge>
                    </div>
                    <div className="h-2 bg-muted rounded-full">
                      <div className="h-2 bg-primary rounded-full" style={{ width: `${item.correlation * 100}%` }} />
                    </div>
                    <p className="text-xs text-muted-foreground mt-2">
                      {item.correlation > 0.8
                        ? `Strong connection between ${item.interest1} and ${item.interest2}`
                        : item.correlation > 0.6
                          ? `Moderate connection between ${item.interest1} and ${item.interest2}`
                          : `Weak connection between ${item.interest1} and ${item.interest2}`}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="timespent" className="h-[350px] relative">
            <Chart
              type="bar"
              data={timeSpentData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  x: {
                    title: {
                      display: true,
                      text: "Interest Topics",
                    },
                  },
                  y: {
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: "Hours per Week",
                    },
                  },
                },
                plugins: {
                  legend: {
                    display: false,
                  },
                },
              }}
            />
            <div className="mt-4 text-sm">
              <div className="flex items-center gap-2">
                <Clock className="h-4 w-4 text-primary" />
                <p className="font-medium">Time Investment:</p>
              </div>
              <p className="text-muted-foreground ml-6">
                Avani spends the most time exploring dinosaurs (4.5 hours/week), followed by space (3.2 hours/week).
              </p>
            </div>
          </TabsContent>

          <TabsContent value="evolution" className="h-[350px] relative">
            <Chart
              type="line"
              data={evolutionData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  x: {
                    title: {
                      display: true,
                      text: "Months",
                    },
                  },
                  y: {
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: "Interest Level (%)",
                    },
                    max: 100,
                  },
                },
                plugins: {
                  tooltip: {
                    mode: "index",
                    intersect: false,
                  },
                },
              }}
            />
            <div className="mt-4 text-sm">
              <div className="flex items-center gap-2">
                <Calendar className="h-4 w-4 text-primary" />
                <p className="font-medium">Interest Evolution:</p>
              </div>
              <p className="text-muted-foreground ml-6">
                Space has shown the most consistent growth over time, while interest in oceans is newly emerging.
              </p>
              <div className="flex justify-end mt-2">
                <Button variant="outline" size="sm">
                  View Detailed Report
                </Button>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
