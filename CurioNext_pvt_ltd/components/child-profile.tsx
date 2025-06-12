"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Textarea } from "@/components/ui/textarea"
import { Switch } from "@/components/ui/switch"
import { Camera, Save } from "lucide-react"

export function ChildProfile() {
  const [formData, setFormData] = useState({
    name: "Avani",
    age: "7",
    learningStyle: "visual",
    languages: "English",
    specialInterests: "Dinosaurs, Space, Animals",
  })

  const handleChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  return (
    <Card>
      <CardContent className="p-6">
        <Tabs defaultValue="profile">
          <TabsList className="mb-6">
            <TabsTrigger value="profile">Profile Information</TabsTrigger>
            <TabsTrigger value="preferences">Learning Preferences</TabsTrigger>
            <TabsTrigger value="privacy">Privacy Settings</TabsTrigger>
          </TabsList>

          <TabsContent value="profile" className="space-y-6">
            <div className="flex flex-col md:flex-row gap-6">
              <div className="flex flex-col items-center gap-4">
                <Avatar className="h-32 w-32">
                  <AvatarImage src="/placeholder.svg?height=128&width=128" alt="Child's avatar" />
                  <AvatarFallback>EM</AvatarFallback>
                </Avatar>
                <Button variant="outline" size="sm">
                  <Camera className="h-4 w-4 mr-2" />
                  Change Photo
                </Button>
              </div>

              <div className="flex-1 grid gap-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">Child's Name</Label>
                    <Input id="name" value={formData.name} onChange={(e) => handleChange("name", e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="age">Age</Label>
                    <Select value={formData.age} onValueChange={(value) => handleChange("age", value)}>
                      <SelectTrigger id="age">
                        <SelectValue placeholder="Select age" />
                      </SelectTrigger>
                      <SelectContent>
                        {Array.from({ length: 10 }, (_, i) => i + 3).map((age) => (
                          <SelectItem key={age} value={age.toString()}>
                            {age} years
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="languages">Languages Spoken</Label>
                  <Input
                    id="languages"
                    value={formData.languages}
                    onChange={(e) => handleChange("languages", e.target.value)}
                    placeholder="e.g., English, Spanish"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="specialInterests">Special Interests or Needs (if applicable)</Label>
                  <Textarea
                    id="specialInterests"
                    value={formData.specialInterests}
                    onChange={(e) => handleChange("specialInterests", e.target.value)}
                    placeholder="Enter any special interests or needs"
                    rows={3}
                  />
                </div>
              </div>
            </div>

            <div className="flex justify-end">
              <Button>
                <Save className="h-4 w-4 mr-2" />
                Save Changes
              </Button>
            </div>
          </TabsContent>

          <TabsContent value="preferences" className="space-y-6">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="learningStyle">Preferred Learning Style</Label>
                <Select value={formData.learningStyle} onValueChange={(value) => handleChange("learningStyle", value)}>
                  <SelectTrigger id="learningStyle">
                    <SelectValue placeholder="Select learning style" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="visual">Visual (learns through seeing)</SelectItem>
                    <SelectItem value="auditory">Auditory (learns through hearing)</SelectItem>
                    <SelectItem value="kinesthetic">Kinesthetic (learns through doing)</SelectItem>
                    <SelectItem value="reading">Reading/Writing</SelectItem>
                    <SelectItem value="multimodal">Multimodal (combination)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-4">
                <h3 className="text-lg font-medium">Content Preferences</h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex items-center justify-between space-x-2">
                    <Label htmlFor="science-content">Science Content</Label>
                    <Switch id="science-content" defaultChecked />
                  </div>

                  <div className="flex items-center justify-between space-x-2">
                    <Label htmlFor="arts-content">Arts & Creativity</Label>
                    <Switch id="arts-content" defaultChecked />
                  </div>

                  <div className="flex items-center justify-between space-x-2">
                    <Label htmlFor="history-content">History & Culture</Label>
                    <Switch id="history-content" defaultChecked />
                  </div>

                  <div className="flex items-center justify-between space-x-2">
                    <Label htmlFor="math-content">Math & Logic</Label>
                    <Switch id="math-content" defaultChecked />
                  </div>

                  <div className="flex items-center justify-between space-x-2">
                    <Label htmlFor="nature-content">Nature & Environment</Label>
                    <Switch id="nature-content" defaultChecked />
                  </div>

                  <div className="flex items-center justify-between space-x-2">
                    <Label htmlFor="social-content">Social & Emotional</Label>
                    <Switch id="social-content" defaultChecked />
                  </div>
                </div>
              </div>
            </div>

            <div className="flex justify-end">
              <Button>
                <Save className="h-4 w-4 mr-2" />
                Save Preferences
              </Button>
            </div>
          </TabsContent>

          <TabsContent value="privacy" className="space-y-6">
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Privacy Controls</h3>

              <div className="space-y-4">
                <div className="flex items-center justify-between space-x-2">
                  <div>
                    <Label htmlFor="show-questions" className="block">
                      Show Specific Questions
                    </Label>
                    <p className="text-sm text-muted-foreground">Allow parents to see specific questions asked</p>
                  </div>
                  <Switch id="show-questions" />
                </div>

                <div className="flex items-center justify-between space-x-2">
                  <div>
                    <Label htmlFor="share-insights" className="block">
                      Share Learning Insights
                    </Label>
                    <p className="text-sm text-muted-foreground">
                      Allow the system to analyze and share learning patterns
                    </p>
                  </div>
                  <Switch id="share-insights" defaultChecked />
                </div>

                <div className="flex items-center justify-between space-x-2">
                  <div>
                    <Label htmlFor="record-audio" className="block">
                      Record Audio Summaries
                    </Label>
                    <p className="text-sm text-muted-foreground">Allow the system to create audio summaries</p>
                  </div>
                  <Switch id="record-audio" defaultChecked />
                </div>

                <div className="flex items-center justify-between space-x-2">
                  <div>
                    <Label htmlFor="location-tracking" className="block">
                      Location Tracking
                    </Label>
                    <p className="text-sm text-muted-foreground">Enable location tracking for safety features</p>
                  </div>
                  <Switch id="location-tracking" defaultChecked />
                </div>
              </div>
            </div>

            <div className="flex justify-end">
              <Button>
                <Save className="h-4 w-4 mr-2" />
                Save Privacy Settings
              </Button>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
