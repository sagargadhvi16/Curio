import type React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

// Sample interest category data
const categories = [
  {
    id: 1,
    name: "How Things Work",
    description: "Engineering, mechanics, physics",
    progress: 85,
    color: "#FF6B6B",
  },
  {
    id: 2,
    name: "Big Questions",
    description: "Philosophy, space, abstract thinking",
    progress: 65,
    color: "#4ECDC4",
  },
  {
    id: 3,
    name: "Creativity & Expression",
    description: "Art, storytelling, music",
    progress: 70,
    color: "#FFD166",
  },
  {
    id: 4,
    name: "World & Cultures",
    description: "History, geography, social trends",
    progress: 50,
    color: "#6A0572",
  },
  {
    id: 5,
    name: "Nature & Life",
    description: "Biology, environment, sustainability",
    progress: 90,
    color: "#1A936F",
  },
  {
    id: 6,
    name: "Self & Emotions",
    description: "Mental well-being, friendships, personal growth",
    progress: 40,
    color: "#C06C84",
  },
]

export function InterestCategories() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Interest Categories</CardTitle>
        <CardDescription>Explore your child's interests across different learning areas</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {categories.map((category) => (
            <div key={category.id} className="space-y-2">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium">{category.name}</h3>
                  <p className="text-sm text-muted-foreground">{category.description}</p>
                </div>
                <span className="text-sm font-medium">{category.progress}%</span>
              </div>
              <Progress
                value={category.progress}
                className="h-2"
                style={
                  {
                    "--progress-background": category.color,
                  } as React.CSSProperties
                }
              />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
