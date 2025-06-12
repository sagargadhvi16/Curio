"use client"

import type React from "react"

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js"
import { Line, Bar, Pie, Doughnut } from "react-chartjs-2"
import { cn } from "@/lib/utils"
import { forwardRef } from "react"

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
)

interface ChartProps {
  type: "line" | "bar" | "pie" | "doughnut"
  data: any
  options?: any
  series?: {
    dataKey: string
    label: string
    color: string
  }[]
  className?: string
}

export const Chart = forwardRef<HTMLCanvasElement, ChartProps>(({ type, data, options, series, className }, ref) => {
  // If series is provided, transform the data for line charts
  let chartData = data

  if (type === "line" && series) {
    chartData = {
      labels: data.map((item: any) => item.name),
      datasets: series.map((serie) => ({
        label: serie.label,
        data: data.map((item: any) => item[serie.dataKey]),
        borderColor: serie.color,
        backgroundColor: `${serie.color}50`,
        fill: true,
        tension: 0.4,
      })),
    }
  }

  switch (type) {
    case "line":
      return <Line ref={ref} data={chartData} options={options} className={className} />
    case "bar":
      return <Bar ref={ref} data={chartData} options={options} className={className} />
    case "pie":
      return <Pie ref={ref} data={chartData} options={options} className={className} />
    case "doughnut":
      return <Doughnut ref={ref} data={chartData} options={options} className={className} />
    default:
      return <Line ref={ref} data={chartData} options={options} className={className} />
  }
})
Chart.displayName = "Chart"

export const ChartContainer = ({ className, children }: { className?: string; children: React.ReactNode }) => {
  return <div className={cn("rounded-md border bg-card text-card-foreground p-4", className)}>{children}</div>
}

export const ChartTooltip = ({ className, children }: { className?: string; children: React.ReactNode }) => {
  return <div className={cn("rounded-md border bg-card text-card-foreground p-4", className)}>{children}</div>
}

export const ChartTooltipContent = ({ className, children }: { className?: string; children: React.ReactNode }) => {
  return <div className={cn("text-sm text-muted-foreground", className)}>{children}</div>
}

export const ChartLegend = ({ className, children }: { className?: string; children: React.ReactNode }) => {
  return <div className={cn("rounded-md border bg-card text-card-foreground p-4", className)}>{children}</div>
}

export const ChartLegendContent = ({ className, children }: { className?: string; children: React.ReactNode }) => {
  return <div className={cn("text-sm text-muted-foreground", className)}>{children}</div>
}

export const ChartStyle = () => {
  return null
}
