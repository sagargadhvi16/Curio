"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Chart } from "@/components/ui/chart"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Filter, Info, ArrowUpRight } from "lucide-react"

// Sample data for study fields
const studyFieldsData = {
  labels: ["Mathematics", "Science", "Literature", "History", "Arts", "Technology", "Languages"],
  datasets: [
    {
      label: "Questions Asked",
      data: [65, 85, 40, 35, 25, 70, 30],
      backgroundColor: [
        "rgba(255, 107, 107, 0.7)",
        "rgba(78, 205, 196, 0.7)",
        "rgba(255, 209, 102, 0.7)",
        "rgba(106, 5, 114, 0.7)",
        "rgba(26, 147, 111, 0.7)",
        "rgba(192, 108, 132, 0.7)",
        "rgba(106, 159, 181, 0.7)",
      ],
      borderColor: [
        "rgb(255, 107, 107)",
        "rgb(78, 205, 196)",
        "rgb(255, 209, 102)",
        "rgb(106, 5, 114)",
        "rgb(26, 147, 111)",
        "rgb(192, 108, 132)",
        "rgb(106, 159, 181)",
      ],
      borderWidth: 1,
    },
  ],
}

// Sample data for subcategories
const subcategoriesData = {
  Mathematics: {
    labels: ["Arithmetic", "Algebra", "Geometry", "Statistics", "Calculus"],
    data: [40, 25, 20, 10, 5],
  },
  Science: {
    labels: ["Biology", "Chemistry", "Physics", "Astronomy", "Earth Science"],
    data: [30, 15, 25, 20, 10],
  },
  Literature: {
    labels: ["Fiction", "Poetry", "Drama", "Non-fiction", "Mythology"],
    data: [45, 15, 10, 20, 10],
  },
  History: {
    labels: ["Ancient", "Medieval", "Modern", "World Wars", "Cultural"],
    data: [25, 20, 30, 15, 10],
  },
  Arts: {
    labels: ["Visual Arts", "Music", "Dance", "Theater", "Crafts"],
    data: [35, 25, 15, 10, 15],
  },
  Technology: {
    labels: ["Computers", "Robotics", "Internet", "Gadgets", "Coding"],
    data: [20, 25, 15, 30, 10],
  },
  Languages: {
    labels: ["English", "Spanish", "French", "Chinese", "Other"],
    data: [50, 20, 15, 10, 5],
  },
}

// Sample data for time trends
const timeTrendData = {
  labels: ["Week 1", "Week 2", "Week 3", "Week 4"],
  datasets: [
    {
      label: "Mathematics",
      data: [20, 25, 30, 35],
      borderColor: "rgb(255, 107, 107)",
      backgroundColor: "rgba(255, 107, 107, 0.1)",
      fill: true,
    },
    {
      label: "Science",
      data: [30, 35, 40, 45],
      borderColor: "rgb(78, 205, 196)",
      backgroundColor: "rgba(78, 205, 196, 0.1)",
      fill: true,
    },
    {
      label: "Technology",
      data: [25, 30, 35, 40],
      borderColor: "rgb(192, 108, 132)",
      backgroundColor: "rgba(192, 108, 132, 0.1)",
      fill: true,
    },
  ],
}

export function StudyFieldAnalysis() {
  const [timeRange, setTimeRange] = useState("month")
  const [selectedField, setSelectedField] = useState<string | null>(null)
  const [showFilters, setShowFilters] = useState(false)

  const handleFieldClick = (field: string) => {
    setSelectedField(field)
  }

  const resetSelection = () => {
    setSelectedField(null)
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Study Field Analysis</CardTitle>
          <CardDescription>Analyze your child's questions across different academic disciplines</CardDescription>
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
          <Button variant="outline" size="icon" onClick={() => setShowFilters(!showFilters)}>
            <Filter className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {showFilters && (
          <div className="mb-4 p-3 bg-muted/30 rounded-md">
            <div className="flex flex-wrap gap-2">
              {studyFieldsData.labels.map((field) => (
                <Badge
                  key={field}
                  variant={selectedField === field ? "default" : "outline"}
                  className="cursor-pointer"
                  onClick={() => handleFieldClick(field)}
                >
                  {field}
                </Badge>
              ))}
              {selectedField && (
                <Button variant="ghost" size="sm" onClick={resetSelection} className="ml-2 h-6">
                  Reset
                </Button>
              )}
            </div>
          </div>
        )}

        <Tabs defaultValue="overview">
          <TabsList className="mb-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="details">Field Details</TabsTrigger>
            <TabsTrigger value="trends">Time Trends</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="h-[350px] relative">
            <Chart
              type="pie"
              data={
                selectedField
                  ? {
                      labels: subcategoriesData[selectedField as keyof typeof subcategoriesData].labels,
                      datasets: [
                        {
                          label: selectedField,
                          data: subcategoriesData[selectedField as keyof typeof subcategoriesData].data,
                          backgroundColor: Array(5).fill(
                            studyFieldsData.datasets[0].backgroundColor[studyFieldsData.labels.indexOf(selectedField)],
                          ),
                          borderColor: Array(5).fill(
                            studyFieldsData.datasets[0].borderColor[studyFieldsData.labels.indexOf(selectedField)],
                          ),
                          borderWidth: 1,
                        },
                      ],
                    }
                  : studyFieldsData
              }
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    position: "right",
                    labels: {
                      boxWidth: 15,
                      padding: 15,
                    },
                  },
                  tooltip: {
                    callbacks: {
                      label: (context) => {
                        const label = context.label || ""
                        const value = context.raw || 0
                        const total = context.dataset.data.reduce((a: number, b: number) => a + b, 0)
                        const percentage = Math.round((value / total) * 100)
                        return `${label}: ${percentage}% (${value} questions)`
                      },
                    },
                  },
                },
              }}
            />
            {selectedField ? (
              <div className="mt-4 text-sm">
                <div className="flex items-center justify-between">
                  <p className="font-medium">Subcategories of {selectedField}</p>
                  <Button variant="ghost" size="sm" onClick={resetSelection} className="h-7 px-2">
                    Back to all fields
                  </Button>
                </div>
              </div>
            ) : (
              <div className="mt-4 text-sm">
                <p className="font-medium">Click on a segment to see subcategories</p>
                <p className="text-muted-foreground">
                  Science and Mathematics are Avani's most explored academic fields this month.
                </p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="details" className="h-[350px] relative">
            <Chart
              type="bar"
              data={{
                labels: studyFieldsData.labels,
                datasets: [
                  {
                    label: "Questions Asked",
                    data: studyFieldsData.datasets[0].data,
                    backgroundColor: studyFieldsData.datasets[0].backgroundColor,
                  },
                ],
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  x: {
                    title: {
                      display: true,
                      text: "Study Fields",
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
                  legend: {
                    display: false,
                  },
                },
              }}
            />
            <div className="mt-4 text-sm">
              <div className="flex items-center gap-2">
                <Info className="h-4 w-4 text-primary" />
                <p className="font-medium">Key Insight:</p>
              </div>
              <p className="text-muted-foreground ml-6">
                Science and Technology questions show the highest engagement, suggesting a strong STEM interest.
              </p>
            </div>
          </TabsContent>

          <TabsContent value="trends" className="h-[350px] relative">
            <Chart
              type="line"
              data={timeTrendData}
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
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Trending Up: Science</p>
                  <p className="text-muted-foreground">Science questions have increased by 50% over the past month.</p>
                </div>
                <Button variant="outline" size="sm" className="h-7">
                  <span>View Full Report</span>
                  <ArrowUpRight className="ml-1 h-3 w-3" />
                </Button>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
