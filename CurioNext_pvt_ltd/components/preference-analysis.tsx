"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Chart } from "@/components/ui/chart"
import { Progress } from "@/components/ui/progress"
import {
  ThumbsUp,
  ThumbsDown,
  Search,
  Filter,
  Download,
  BarChart3,
  PieChart,
  Smile,
  Frown,
  Meh,
  TrendingUp,
  Calendar,
  RefreshCw,
} from "lucide-react"

// Sample data for likes and dislikes
const preferencesData = {
  likes: [
    { topic: "Dinosaurs", frequency: 85, sentiment: 0.92, questions: 42, trend: "up" },
    { topic: "Space Exploration", frequency: 78, sentiment: 0.88, questions: 36, trend: "up" },
    { topic: "Painting", frequency: 75, sentiment: 0.9, questions: 33, trend: "up" },
    { topic: "Volcanoes", frequency: 72, sentiment: 0.85, questions: 31, trend: "stable" },
    { topic: "Animals", frequency: 68, sentiment: 0.82, questions: 29, trend: "up" },
    { topic: "Science Experiments", frequency: 65, sentiment: 0.79, questions: 27, trend: "up" },
    { topic: "Robots", frequency: 60, sentiment: 0.76, questions: 25, trend: "up" },
    { topic: "Planets", frequency: 58, sentiment: 0.75, questions: 24, trend: "stable" },
    { topic: "Fossils", frequency: 55, sentiment: 0.72, questions: 22, trend: "down" },
    { topic: "Oceans", frequency: 52, sentiment: 0.7, questions: 21, trend: "up" },
    { topic: "Weather", frequency: 48, sentiment: 0.68, questions: 19, trend: "stable" },
  ],
  dislikes: [
    { topic: "Scary Stories", frequency: 30, sentiment: -0.75, questions: 12, trend: "stable" },
    { topic: "Spiders", frequency: 25, sentiment: -0.7, questions: 10, trend: "down" },
    { topic: "Loud Noises", frequency: 20, sentiment: -0.65, questions: 8, trend: "stable" },
    { topic: "Dark Rooms", frequency: 18, sentiment: -0.6, questions: 7, trend: "down" },
    { topic: "Bitter Foods", frequency: 15, sentiment: -0.55, questions: 6, trend: "stable" },
  ],
  neutral: [
    { topic: "Mathematics", frequency: 45, sentiment: 0.1, questions: 18, trend: "up" },
    { topic: "History", frequency: 40, sentiment: 0.05, questions: 16, trend: "stable" },
    { topic: "Geography", frequency: 35, sentiment: 0.0, questions: 14, trend: "stable" },
    { topic: "Languages", frequency: 30, sentiment: -0.05, questions: 12, trend: "down" },
    { topic: "Art", frequency: 25, sentiment: 0.15, questions: 10, trend: "up" },
  ],
}

// Sample recent questions with sentiment analysis
const recentQuestions = [
  {
    id: 1,
    question: "Why did dinosaurs become extinct?",
    timestamp: new Date(Date.now() - 1000 * 60 * 30), // 30 minutes ago
    topics: ["Dinosaurs", "Extinction", "Science"],
    sentiment: 0.6,
    keywords: ["dinosaurs", "extinct", "meteor"],
    inferred_preference: "like",
  },
  {
    id: 2,
    question: "How do volcanoes erupt?",
    timestamp: new Date(Date.now() - 1000 * 60 * 120), // 2 hours ago
    topics: ["Volcanoes", "Geology", "Science"],
    sentiment: 0.8,
    keywords: ["volcanoes", "erupt", "lava"],
    inferred_preference: "like",
  },
  {
    id: 3,
    question: "Are there spiders in space?",
    timestamp: new Date(Date.now() - 1000 * 60 * 180), // 3 hours ago
    topics: ["Space", "Spiders", "Animals"],
    sentiment: -0.3,
    keywords: ["spiders", "space", "scary"],
    inferred_preference: "mixed",
  },
  {
    id: 4,
    question: "Why do I have to eat vegetables?",
    timestamp: new Date(Date.now() - 1000 * 60 * 240), // 4 hours ago
    topics: ["Food", "Health", "Vegetables"],
    sentiment: -0.5,
    keywords: ["vegetables", "eat", "healthy"],
    inferred_preference: "dislike",
  },
  {
    id: 5,
    question: "How do rockets fly to the moon?",
    timestamp: new Date(Date.now() - 1000 * 60 * 300), // 5 hours ago
    topics: ["Space", "Rockets", "Moon"],
    sentiment: 0.9,
    keywords: ["rockets", "moon", "fly"],
    inferred_preference: "like",
  },
]

// Sample data for sentiment over time
const sentimentTimeData = {
  labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"],
  datasets: [
    {
      label: "Positive Sentiment",
      data: [65, 70, 75, 72, 78, 82, 88],
      borderColor: "rgb(34, 197, 94)",
      backgroundColor: "rgba(34, 197, 94, 0.1)",
      fill: true,
    },
    {
      label: "Neutral Sentiment",
      data: [25, 20, 18, 22, 15, 12, 8],
      borderColor: "rgb(59, 130, 246)",
      backgroundColor: "rgba(59, 130, 246, 0.1)",
      fill: true,
    },
    {
      label: "Negative Sentiment",
      data: [10, 10, 7, 6, 7, 6, 4],
      borderColor: "rgb(239, 68, 68)",
      backgroundColor: "rgba(239, 68, 68, 0.1)",
      fill: true,
    },
  ],
}

// Sample data for topic distribution
const topicDistributionData = {
  labels: ["Science", "Animals", "Space", "Art & Painting", "Nature", "Technology", "Others"],
  datasets: [
    {
      label: "Topic Distribution",
      data: [32, 18, 15, 14, 10, 8, 3],
      backgroundColor: [
        "rgba(34, 197, 94, 0.7)",
        "rgba(59, 130, 246, 0.7)",
        "rgba(139, 92, 246, 0.7)",
        "rgba(236, 72, 153, 0.7)",
        "rgba(249, 115, 22, 0.7)",
        "rgba(239, 68, 68, 0.7)",
        "rgba(156, 163, 175, 0.7)",
      ],
      borderColor: [
        "rgb(34, 197, 94)",
        "rgb(59, 130, 246)",
        "rgb(139, 92, 246)",
        "rgb(236, 72, 153)",
        "rgb(249, 115, 22)",
        "rgb(239, 68, 68)",
        "rgb(156, 163, 175)",
      ],
      borderWidth: 1,
    },
  ],
}

// Sample data for sentiment distribution
const sentimentDistributionData = {
  labels: ["Positive", "Neutral", "Negative"],
  datasets: [
    {
      label: "Sentiment Distribution",
      data: [65, 25, 10],
      backgroundColor: ["rgba(34, 197, 94, 0.7)", "rgba(59, 130, 246, 0.7)", "rgba(239, 68, 68, 0.7)"],
      borderColor: ["rgb(34, 197, 94)", "rgb(59, 130, 246)", "rgb(239, 68, 68)"],
      borderWidth: 1,
    },
  ],
}

export function PreferenceAnalysis() {
  const [timeRange, setTimeRange] = useState("month")
  const [activeTab, setActiveTab] = useState("overview")
  const [showFilters, setShowFilters] = useState(false)
  const [topicFilter, setTopicFilter] = useState<string | null>(null)
  const [sentimentFilter, setSentimentFilter] = useState<string | null>(null)
  const [isRefreshing, setIsRefreshing] = useState(false)

  const handleRefresh = () => {
    setIsRefreshing(true)
    // Simulate data refresh
    setTimeout(() => {
      setIsRefreshing(false)
    }, 1500)
  }

  const getSentimentIcon = (sentiment: number) => {
    if (sentiment > 0.3) return <Smile className="h-4 w-4 text-green-500" />
    if (sentiment < -0.3) return <Frown className="h-4 w-4 text-red-500" />
    return <Meh className="h-4 w-4 text-blue-500" />
  }

  const getSentimentColor = (sentiment: number) => {
    if (sentiment > 0.3) return "bg-green-500"
    if (sentiment < -0.3) return "bg-red-500"
    return "bg-blue-500"
  }

  const getTrendIcon = (trend: string) => {
    if (trend === "up") return <TrendingUp className="h-3 w-3 text-green-500" />
    if (trend === "down") return <TrendingUp className="h-3 w-3 text-red-500 rotate-180" />
    return <TrendingUp className="h-3 w-3 text-blue-500 rotate-90" />
  }

  const formatTimestamp = (date: Date) => {
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.round(diffMs / 60000)

    if (diffMins < 60) {
      return `${diffMins} minute${diffMins !== 1 ? "s" : ""} ago`
    }

    const diffHours = Math.floor(diffMins / 60)
    if (diffHours < 24) {
      return `${diffHours} hour${diffHours !== 1 ? "s" : ""} ago`
    }

    const diffDays = Math.floor(diffHours / 24)
    return `${diffDays} day${diffDays !== 1 ? "s" : ""} ago`
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Preference Analysis</CardTitle>
            <CardDescription>Analysis of Avani's questions to infer likes, dislikes, and interests</CardDescription>
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
            <Button variant="outline" size="icon" onClick={handleRefresh} disabled={isRefreshing}>
              {isRefreshing ? <RefreshCw className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            </Button>
            <Button variant="outline" size="icon">
              <Download className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {showFilters && (
            <div className="mb-6 p-4 bg-muted/30 rounded-lg space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Topic Filter</label>
                  <Select value={topicFilter || ""} onValueChange={(value) => setTopicFilter(value || null)}>
                    <SelectTrigger>
                      <SelectValue placeholder="All Topics" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Topics</SelectItem>
                      <SelectItem value="science">Science</SelectItem>
                      <SelectItem value="animals">Animals</SelectItem>
                      <SelectItem value="space">Space</SelectItem>
                      <SelectItem value="nature">Nature</SelectItem>
                      <SelectItem value="technology">Technology</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Sentiment Filter</label>
                  <Select value={sentimentFilter || ""} onValueChange={(value) => setSentimentFilter(value || null)}>
                    <SelectTrigger>
                      <SelectValue placeholder="All Sentiments" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Sentiments</SelectItem>
                      <SelectItem value="positive">Positive</SelectItem>
                      <SelectItem value="neutral">Neutral</SelectItem>
                      <SelectItem value="negative">Negative</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="flex justify-end">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setTopicFilter("all")
                    setSentimentFilter("all")
                  }}
                >
                  Clear Filters
                </Button>
              </div>
            </div>
          )}

          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="mb-4">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="likes">Likes</TabsTrigger>
              <TabsTrigger value="dislikes">Dislikes</TabsTrigger>
              <TabsTrigger value="topics">Topic Analysis</TabsTrigger>
              <TabsTrigger value="trends">Trends</TabsTrigger>
            </TabsList>

            {/* Overview Tab */}
            <TabsContent value="overview">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-medium mb-4">Sentiment Distribution</h3>
                  <div className="h-[300px]">
                    <Chart
                      type="doughnut"
                      data={sentimentDistributionData}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: {
                            position: "right",
                          },
                          tooltip: {
                            callbacks: {
                              label: (context) => {
                                const label = context.label || ""
                                const value = context.raw || 0
                                const total = context.dataset.data.reduce((a: number, b: number) => a + b, 0)
                                const percentage = Math.round((value / total) * 100)
                                return `${label}: ${percentage}%`
                              },
                            },
                          },
                        },
                      }}
                    />
                  </div>
                  <div className="mt-4 text-sm">
                    <p className="font-medium">Key Insight:</p>
                    <p className="text-muted-foreground">
                      Avani's questions show predominantly positive sentiment (65%), indicating strong engagement and
                      curiosity.
                    </p>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-medium mb-4">Topic Distribution</h3>
                  <div className="h-[300px]">
                    <Chart
                      type="pie"
                      data={topicDistributionData}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: {
                            position: "right",
                          },
                          tooltip: {
                            callbacks: {
                              label: (context) => {
                                const label = context.label || ""
                                const value = context.raw || 0
                                const total = context.dataset.data.reduce((a: number, b: number) => a + b, 0)
                                const percentage = Math.round((value / total) * 100)
                                return `${label}: ${percentage}%`
                              },
                            },
                          },
                        },
                      }}
                    />
                  </div>
                  <div className="mt-4 text-sm">
                    <p className="font-medium">Key Insight:</p>
                    <p className="text-muted-foreground">
                      Science topics dominate Avani's questions (35%), followed by animals (20%) and space (15%).
                    </p>
                  </div>
                </div>
              </div>

              <div className="mt-6">
                <h3 className="text-lg font-medium mb-4">Top Preferences Summary</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <ThumbsUp className="h-5 w-5 text-green-500" />
                      <h4 className="font-medium">Top Likes</h4>
                    </div>
                    <div className="space-y-3">
                      {preferencesData.likes.slice(0, 5).map((item, index) => (
                        <div key={index} className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium">{item.topic}</span>
                            {getTrendIcon(item.trend)}
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge variant="outline" className="text-xs">
                              {item.questions} questions
                            </Badge>
                            <div className="w-24 h-2 bg-muted rounded-full overflow-hidden">
                              <div
                                className="h-full bg-green-500 rounded-full"
                                style={{ width: `${item.frequency}%` }}
                              />
                            </div>
                            <span className="text-xs font-medium">{item.frequency}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <ThumbsDown className="h-5 w-5 text-red-500" />
                      <h4 className="font-medium">Top Dislikes</h4>
                    </div>
                    <div className="space-y-3">
                      {preferencesData.dislikes.slice(0, 5).map((item, index) => (
                        <div key={index} className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium">{item.topic}</span>
                            {getTrendIcon(item.trend)}
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge variant="outline" className="text-xs">
                              {item.questions} questions
                            </Badge>
                            <div className="w-24 h-2 bg-muted rounded-full overflow-hidden">
                              <div className="h-full bg-red-500 rounded-full" style={{ width: `${item.frequency}%` }} />
                            </div>
                            <span className="text-xs font-medium">{item.frequency}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* Likes Tab */}
            <TabsContent value="likes">
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-medium">Identified Likes</h3>
                  <div className="flex items-center gap-2">
                    <Search className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">
                      {preferencesData.likes.length} topics identified
                    </span>
                  </div>
                </div>

                <div className="space-y-4">
                  {preferencesData.likes.map((item, index) => (
                    <div key={index} className="border rounded-lg p-4">
                      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                        <div className="flex items-start gap-3">
                          <div className="p-2 rounded-full bg-green-100 dark:bg-green-900">
                            <ThumbsUp className="h-5 w-5 text-green-500" />
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                              <h4 className="font-medium">{item.topic}</h4>
                              <Badge className="bg-green-500">{item.frequency}%</Badge>
                            </div>
                            <div className="flex items-center gap-2 mt-1">
                              <Badge variant="outline" className="text-xs">
                                {item.questions} questions
                              </Badge>
                              <div className="flex items-center gap-1">
                                <span className="text-xs text-muted-foreground">Sentiment:</span>
                                <div className="flex items-center">
                                  {getSentimentIcon(item.sentiment)}
                                  <span className="text-xs ml-1">{item.sentiment.toFixed(2)}</span>
                                </div>
                              </div>
                              <div className="flex items-center gap-1">
                                <span className="text-xs text-muted-foreground">Trend:</span>
                                {getTrendIcon(item.trend)}
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="w-full md:w-1/3">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs text-muted-foreground">Frequency</span>
                            <span className="text-xs font-medium">{item.frequency}%</span>
                          </div>
                          <Progress value={item.frequency} className="h-2" />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </TabsContent>

            {/* Dislikes Tab */}
            <TabsContent value="dislikes">
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-medium">Identified Dislikes</h3>
                  <div className="flex items-center gap-2">
                    <Search className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">
                      {preferencesData.dislikes.length} topics identified
                    </span>
                  </div>
                </div>

                <div className="space-y-4">
                  {preferencesData.dislikes.map((item, index) => (
                    <div key={index} className="border rounded-lg p-4">
                      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                        <div className="flex items-start gap-3">
                          <div className="p-2 rounded-full bg-red-100 dark:bg-red-900">
                            <ThumbsDown className="h-5 w-5 text-red-500" />
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                              <h4 className="font-medium">{item.topic}</h4>
                              <Badge className="bg-red-500">{item.frequency}%</Badge>
                            </div>
                            <div className="flex items-center gap-2 mt-1">
                              <Badge variant="outline" className="text-xs">
                                {item.questions} questions
                              </Badge>
                              <div className="flex items-center gap-1">
                                <span className="text-xs text-muted-foreground">Sentiment:</span>
                                <div className="flex items-center">
                                  {getSentimentIcon(item.sentiment)}
                                  <span className="text-xs ml-1">{item.sentiment.toFixed(2)}</span>
                                </div>
                              </div>
                              <div className="flex items-center gap-1">
                                <span className="text-xs text-muted-foreground">Trend:</span>
                                {getTrendIcon(item.trend)}
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="w-full md:w-1/3">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs text-muted-foreground">Frequency</span>
                            <span className="text-xs font-medium">{item.frequency}%</span>
                          </div>
                          <Progress value={item.frequency} className="h-2" />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-4">
                  <h3 className="text-lg font-medium mb-4">Neutral Topics</h3>
                  <div className="space-y-4">
                    {preferencesData.neutral.map((item, index) => (
                      <div key={index} className="border rounded-lg p-4">
                        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                          <div className="flex items-start gap-3">
                            <div className="p-2 rounded-full bg-blue-100 dark:bg-blue-900">
                              <Meh className="h-5 w-5 text-blue-500" />
                            </div>
                            <div>
                              <div className="flex items-center gap-2">
                                <h4 className="font-medium">{item.topic}</h4>
                                <Badge variant="secondary">{item.frequency}%</Badge>
                              </div>
                              <div className="flex items-center gap-2 mt-1">
                                <Badge variant="outline" className="text-xs">
                                  {item.questions} questions
                                </Badge>
                                <div className="flex items-center gap-1">
                                  <span className="text-xs text-muted-foreground">Sentiment:</span>
                                  <div className="flex items-center">
                                    {getSentimentIcon(item.sentiment)}
                                    <span className="text-xs ml-1">{item.sentiment.toFixed(2)}</span>
                                  </div>
                                </div>
                                <div className="flex items-center gap-1">
                                  <span className="text-xs text-muted-foreground">Trend:</span>
                                  {getTrendIcon(item.trend)}
                                </div>
                              </div>
                            </div>
                          </div>
                          <div className="w-full md:w-1/3">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs text-muted-foreground">Frequency</span>
                              <span className="text-xs font-medium">{item.frequency}%</span>
                            </div>
                            <Progress value={item.frequency} className="h-2" />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* Topic Analysis Tab - Privacy-focused alternative to showing actual questions */}
            <TabsContent value="topics">
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-medium">Topic Analysis</h3>
                  <div className="flex items-center gap-2">
                    <Search className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">15 topics analyzed</span>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="border rounded-lg p-4">
                    <div className="flex items-start gap-3">
                      <div className="p-2 rounded-full bg-green-100 dark:bg-green-900">
                        <ThumbsUp className="h-5 w-5 text-green-500" />
                      </div>
                      <div className="flex-1">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                          <h4 className="font-medium">Dinosaurs</h4>
                          <div className="flex items-center gap-2">
                            <Badge className="bg-green-500">High Interest</Badge>
                            <span className="text-xs text-muted-foreground">42 related topics detected</span>
                          </div>
                        </div>
                        <div className="mt-2 flex flex-wrap gap-2">
                          <Badge variant="outline" className="text-xs">
                            Extinction
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            Fossils
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            Prehistoric
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            T-Rex
                          </Badge>
                        </div>
                        <div className="mt-2 flex items-center gap-4">
                          <div className="flex items-center gap-1">
                            <span className="text-xs text-muted-foreground">Sentiment:</span>
                            <div className="flex items-center">
                              <Smile className="h-4 w-4 text-green-500" />
                              <span className="text-xs ml-1">0.92</span>
                            </div>
                          </div>
                          <div className="flex items-center gap-1">
                            <span className="text-xs text-muted-foreground">Trend:</span>
                            <TrendingUp className="h-3 w-3 text-green-500" />
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="border rounded-lg p-4">
                    <div className="flex items-start gap-3">
                      <div className="p-2 rounded-full bg-green-100 dark:bg-green-900">
                        <ThumbsUp className="h-5 w-5 text-green-500" />
                      </div>
                      <div className="flex-1">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                          <h4 className="font-medium">Space Exploration</h4>
                          <div className="flex items-center gap-2">
                            <Badge className="bg-green-500">High Interest</Badge>
                            <span className="text-xs text-muted-foreground">36 related topics detected</span>
                          </div>
                        </div>
                        <div className="mt-2 flex flex-wrap gap-2">
                          <Badge variant="outline" className="text-xs">
                            Planets
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            Rockets
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            Astronauts
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            Moon
                          </Badge>
                        </div>
                        <div className="mt-2 flex items-center gap-4">
                          <div className="flex items-center gap-1">
                            <span className="text-xs text-muted-foreground">Sentiment:</span>
                            <div className="flex items-center">
                              <Smile className="h-4 w-4 text-green-500" />
                              <span className="text-xs ml-1">0.88</span>
                            </div>
                          </div>
                          <div className="flex items-center gap-1">
                            <span className="text-xs text-muted-foreground">Trend:</span>
                            <TrendingUp className="h-3 w-3 text-green-500" />
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="border rounded-lg p-4">
                    <div className="flex items-start gap-3">
                      <div className="p-2 rounded-full bg-green-100 dark:bg-green-900">
                        <ThumbsUp className="h-5 w-5 text-green-500" />
                      </div>
                      <div className="flex-1">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                          <h4 className="font-medium">Painting</h4>
                          <div className="flex items-center gap-2">
                            <Badge className="bg-green-500">High Interest</Badge>
                            <span className="text-xs text-muted-foreground">33 related topics detected</span>
                          </div>
                        </div>
                        <div className="mt-2 flex flex-wrap gap-2">
                          <Badge variant="outline" className="text-xs">
                            Colors
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            Canvas
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            Brushes
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            Techniques
                          </Badge>
                        </div>
                        <div className="mt-2 flex items-center gap-4">
                          <div className="flex items-center gap-1">
                            <span className="text-xs text-muted-foreground">Sentiment:</span>
                            <div className="flex items-center">
                              <Smile className="h-4 w-4 text-green-500" />
                              <span className="text-xs ml-1">0.90</span>
                            </div>
                          </div>
                          <div className="flex items-center gap-1">
                            <span className="text-xs text-muted-foreground">Trend:</span>
                            <TrendingUp className="h-3 w-3 text-green-500" />
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="border rounded-lg p-4">
                    <div className="flex items-start gap-3">
                      <div className="p-2 rounded-full bg-red-100 dark:bg-red-900">
                        <ThumbsDown className="h-5 w-5 text-red-500" />
                      </div>
                      <div className="flex-1">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                          <h4 className="font-medium">Scary Stories</h4>
                          <div className="flex items-center gap-2">
                            <Badge className="bg-red-500">Low Interest</Badge>
                            <span className="text-xs text-muted-foreground">12 related topics detected</span>
                          </div>
                        </div>
                        <div className="mt-2 flex flex-wrap gap-2">
                          <Badge variant="outline" className="text-xs">
                            Ghosts
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            Monsters
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            Dark
                          </Badge>
                        </div>
                        <div className="mt-2 flex items-center gap-4">
                          <div className="flex items-center gap-1">
                            <span className="text-xs text-muted-foreground">Sentiment:</span>
                            <div className="flex items-center">
                              <Frown className="h-4 w-4 text-red-500" />
                              <span className="text-xs ml-1">-0.75</span>
                            </div>
                          </div>
                          <div className="flex items-center gap-1">
                            <span className="text-xs text-muted-foreground">Trend:</span>
                            <TrendingUp className="h-3 w-3 text-blue-500 rotate-90" />
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* Trends Tab */}
            <TabsContent value="trends">
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium mb-4">Sentiment Trends Over Time</h3>
                  <div className="h-[350px]">
                    <Chart
                      type="line"
                      data={sentimentTimeData}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                          x: {
                            title: {
                              display: true,
                              text: "Month",
                            },
                          },
                          y: {
                            beginAtZero: true,
                            title: {
                              display: true,
                              text: "Percentage (%)",
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
                  </div>
                  <div className="mt-4 text-sm">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-primary" />
                      <p className="font-medium">Key Insight:</p>
                    </div>
                    <p className="text-muted-foreground ml-6">
                      Positive sentiment has been steadily increasing over the past 7 months, with a notable spike in
                      July coinciding with Avani's new interest in painting and art.
                    </p>
                  </div>
                </div>

                <div className="mt-6">
                  <h3 className="text-lg font-medium mb-4">Emerging Interests</h3>
                  <div className="space-y-4">
                    <div className="border rounded-lg p-4">
                      <div className="flex items-start gap-3">
                        <div className="p-2 rounded-full bg-primary/10">
                          <TrendingUp className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <h4 className="font-medium">Painting & Art</h4>
                          <p className="text-sm text-muted-foreground mt-1">
                            Questions about painting techniques, colors, and famous artists have emerged as a new
                            interest, showing a 45% increase in the last week.
                          </p>
                          <div className="mt-2 flex items-center gap-2">
                            <Badge variant="outline" className="text-xs">
                              33 new questions
                            </Badge>
                            <Badge className="bg-green-500">Strong positive sentiment</Badge>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="border rounded-lg p-4">
                      <div className="flex items-start gap-3">
                        <div className="p-2 rounded-full bg-primary/10">
                          <TrendingUp className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <h4 className="font-medium">Astronomy</h4>
                          <p className="text-sm text-muted-foreground mt-1">
                            Questions about stars, planets, and space exploration have increased by 35% in the last
                            month.
                          </p>
                          <div className="mt-2 flex items-center gap-2">
                            <Badge variant="outline" className="text-xs">
                              18 new questions
                            </Badge>
                            <Badge className="bg-green-500">Strong positive sentiment</Badge>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="border rounded-lg p-4">
                      <div className="flex items-start gap-3">
                        <div className="p-2 rounded-full bg-primary/10">
                          <TrendingUp className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <h4 className="font-medium">Robotics</h4>
                          <p className="text-sm text-muted-foreground mt-1">
                            Interest in how robots work and are built has grown by 28% since last quarter.
                          </p>
                          <div className="mt-2 flex items-center gap-2">
                            <Badge variant="outline" className="text-xs">
                              12 new questions
                            </Badge>
                            <Badge className="bg-green-500">Strong positive sentiment</Badge>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="border rounded-lg p-4">
                      <div className="flex items-start gap-3">
                        <div className="p-2 rounded-full bg-primary/10">
                          <Calendar className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <h4 className="font-medium">Seasonal Patterns</h4>
                          <p className="text-sm text-muted-foreground mt-1">
                            Questions about nature and animals tend to increase during spring months, while
                            space-related questions peak in winter.
                          </p>
                          <div className="mt-2 flex items-center gap-2">
                            <Badge variant="outline" className="text-xs">
                              Pattern detected
                            </Badge>
                            <Badge variant="secondary">Moderate confidence</Badge>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>How Preference Analysis Works</CardTitle>
          <CardDescription>Understanding how we analyze Avani's questions to infer preferences</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="border rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Search className="h-5 w-5 text-primary" />
                  <h3 className="font-medium">Keyword Extraction</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  We analyze each question to identify key topics, subjects, and themes using natural language
                  processing.
                </p>
              </div>

              <div className="border rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <BarChart3 className="h-5 w-5 text-primary" />
                  <h3 className="font-medium">Frequency Analysis</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  Topics that appear frequently in questions are tracked and ranked to identify patterns of interest.
                </p>
              </div>

              <div className="border rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <PieChart className="h-5 w-5 text-primary" />
                  <h3 className="font-medium">Sentiment Analysis</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  We evaluate the emotional tone of questions to determine if Avani has positive, negative, or neutral
                  feelings about topics.
                </p>
              </div>
            </div>

            <div className="mt-4 p-4 bg-muted/30 rounded-lg">
              <p className="text-sm">
                <strong>Privacy Note:</strong> All analysis is performed on-device and no raw question data is stored
                long-term. Only aggregated insights are saved to protect Avani's privacy while providing valuable
                insights.
                <strong className="block mt-2">
                  For privacy protection, we never display the actual questions asked by your child.
                </strong>
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
