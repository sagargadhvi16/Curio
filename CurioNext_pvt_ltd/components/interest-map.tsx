"use client"

import { WeeklySummary } from "@/app/page"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Chart } from "@/components/ui/chart"
import { useMemo } from "react"


function convertToInterestData(data: WeeklySummary['interest_distribution']) {
  const colors = ["#FF6B6B", "#4ECDC4", "#FFD166", "#6A0572", "#1A936F", "#C06C84"];
  return Object.keys(data).map((key, index) => ({
    id: index + 1,
    name: key,
    color: colors[index] ?? colors[0],
    current: data[key],
  }))
}


type props = {
  data: WeeklySummary['interest_distribution']
}

export function InterestMap( { data } : props) {
  // Prepare data for pie chart
  const pieChartData = useMemo(() => {
    const interestData = convertToInterestData(data)
    return {labels: interestData.map((item) => item.name),
    datasets: [
      {
        data: interestData.map((item) => item.current),
        backgroundColor: interestData.map((item) => item.color),
        borderColor: interestData.map((item) => item.color),
        borderWidth: 1,
      },
    ],
  }}, [data])

  return (
    <Card className="col-span-1 md:col-span-1">
      <CardHeader>
        <div>
          <CardTitle>Interest Distribution</CardTitle>
          <CardDescription>This Week's Curiosity Distribution</CardDescription>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-[350px] relative">
          <Chart
            type="pie"
            data={{
              labels: pieChartData.labels,
              datasets: pieChartData.datasets,
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: {
                  position: "right",
                },
                tooltip: {
                  callbacks: {
                    label: (context: { label: string; raw: number }) => {
                      const label = context.label || ""
                      const value = context.raw || 0
                      return `${label}: ${value}%`
                    },
                  },
                },
              },
            }}
          />
        </div>
      </CardContent>
    </Card>
  )
}
