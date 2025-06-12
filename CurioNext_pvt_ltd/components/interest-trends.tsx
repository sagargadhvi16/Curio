"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Chart } from "@/components/ui/chart"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useState } from "react"

// Sample trend data
const trendData = {
  "1 Month": [
    { name: "Week 1", "How Things Work": 20, "Nature & Life": 30, "Big Questions": 10 },
    { name: "Week 2", "How Things Work": 25, "Nature & Life": 35, "Big Questions": 15 },
    { name: "Week 3", "How Things Work": 30, "Nature & Life": 40, "Big Questions": 20 },
    { name: "Week 4", "How Things Work": 40, "Nature & Life": 45, "Big Questions": 25 },
  ],
  "3 Months": [
    { name: "Jan", "How Things Work": 15, "Nature & Life": 20, "Big Questions": 5 },
    { name: "Feb", "How Things Work": 25, "Nature & Life": 30, "Big Questions": 15 },
    { name: "Mar", "How Things Work": 40, "Nature & Life": 45, "Big Questions": 25 },
  ],
  "6 Months": [
    { name: "Oct", "How Things Work": 10, "Nature & Life": 15, "Big Questions": 5 },
    { name: "Nov", "How Things Work": 15, "Nature & Life": 20, "Big Questions": 10 },
    { name: "Dec", "How Things Work": 20, "Nature & Life": 25, "Big Questions": 15 },
    { name: "Jan", "How Things Work": 25, "Nature & Life": 30, "Big Questions": 20 },
    { name: "Feb", "How Things Work": 35, "Nature & Life": 35, "Big Questions": 20 },
    { name: "Mar", "How Things Work": 40, "Nature & Life": 45, "Big Questions": 25 },
  ],
}

export function InterestTrends() {
  const [timeRange, setTimeRange] = useState("1 Month")

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Interest Trends</CardTitle>
          <CardDescription>Track how your child's interests have evolved over time</CardDescription>
        </div>
        <Select value={timeRange} onValueChange={setTimeRange}>
          <SelectTrigger className="w-[120px]">
            <SelectValue placeholder="Select range" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="1 Month">1 Month</SelectItem>
            <SelectItem value="3 Months">3 Months</SelectItem>
            <SelectItem value="6 Months">6 Months</SelectItem>
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent>
        <div className="h-[300px]">
          <Chart
            type="line"
            data={trendData[timeRange as keyof typeof trendData]}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                x: {
                  grid: {
                    display: false,
                  },
                },
                y: {
                  beginAtZero: true,
                  max: 50,
                  ticks: {
                    stepSize: 10,
                  },
                },
              },
              plugins: {
                legend: {
                  position: "top",
                },
                tooltip: {
                  mode: "index",
                  intersect: false,
                },
              },
            }}
            series={[
              {
                dataKey: "How Things Work",
                label: "How Things Work",
                color: "#FF6B6B",
              },
              {
                dataKey: "Nature & Life",
                label: "Nature & Life",
                color: "#1A936F",
              },
              {
                dataKey: "Big Questions",
                label: "Big Questions",
                color: "#4ECDC4",
              },
            ]}
          />
        </div>
      </CardContent>
    </Card>
  )
}
