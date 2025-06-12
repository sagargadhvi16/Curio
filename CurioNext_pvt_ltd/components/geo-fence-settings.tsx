"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Slider } from "@/components/ui/slider"
import { MapPin, Plus, Trash2, Save, School, Home, Library, Edit } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

// Sample geo fence data
const initialGeoFences = [
  {
    id: 1,
    name: "School",
    address: "123 Education Ave, Learning City",
    coordinates: { lat: 37.7749, lng: -122.4194 },
    radius: 200,
    enabled: true,
    icon: School,
    schedule: "Mon-Fri, 8:00 AM - 3:00 PM",
  },
  {
    id: 2,
    name: "Library",
    address: "456 Book St, Reading Town",
    coordinates: { lat: 37.7749, lng: -122.4194 },
    radius: 100,
    enabled: true,
    icon: Library,
    schedule: "All days, 9:00 AM - 7:00 PM",
  },
  {
    id: 3,
    name: "Home Study Area",
    address: "789 Home Lane, Family City",
    coordinates: { lat: 37.7749, lng: -122.4194 },
    radius: 50,
    enabled: false,
    icon: Home,
    schedule: "Mon-Fri, 4:00 PM - 8:00 PM",
  },
]

export function GeoFenceSettings() {
  const [geoFences, setGeoFences] = useState(initialGeoFences)
  const [newFence, setNewFence] = useState({
    name: "",
    address: "",
    radius: 100,
    enabled: true,
    schedule: "All days, All hours",
  })
  const [editingFence, setEditingFence] = useState<number | null>(null)
  const [activeTab, setActiveTab] = useState("list")

  const handleToggle = (id: number) => {
    setGeoFences(geoFences.map((fence) => (fence.id === id ? { ...fence, enabled: !fence.enabled } : fence)))
  }

  const handleDelete = (id: number) => {
    setGeoFences(geoFences.filter((fence) => fence.id !== id))
  }

  const handleEdit = (id: number) => {
    const fence = geoFences.find((f) => f.id === id)
    if (fence) {
      setNewFence({
        name: fence.name,
        address: fence.address,
        radius: fence.radius,
        enabled: fence.enabled,
        schedule: fence.schedule || "All days, All hours",
      })
      setEditingFence(id)
      setActiveTab("add")
    }
  }

  const handleSave = () => {
    if (editingFence) {
      setGeoFences(
        geoFences.map((fence) =>
          fence.id === editingFence
            ? {
                ...fence,
                name: newFence.name,
                address: newFence.address,
                radius: newFence.radius,
                enabled: newFence.enabled,
                schedule: newFence.schedule,
              }
            : fence,
        ),
      )
      setEditingFence(null)
    } else {
      const newId = Math.max(0, ...geoFences.map((f) => f.id)) + 1
      setGeoFences([
        ...geoFences,
        {
          id: newId,
          name: newFence.name,
          address: newFence.address,
          coordinates: { lat: 37.7749, lng: -122.4194 }, // Placeholder coordinates
          radius: newFence.radius,
          enabled: newFence.enabled,
          icon: MapPin,
          schedule: newFence.schedule,
        },
      ])
    }

    // Reset form
    setNewFence({
      name: "",
      address: "",
      radius: 100,
      enabled: true,
      schedule: "All days, All hours",
    })
    setActiveTab("list")
  }

  const handleCancel = () => {
    setNewFence({
      name: "",
      address: "",
      radius: 100,
      enabled: true,
      schedule: "All days, All hours",
    })
    setEditingFence(null)
    setActiveTab("list")
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>GEO Fence Settings</CardTitle>
            <CardDescription>
              Define areas where the device's question-answering functionality is disabled
            </CardDescription>
          </div>
          <Button variant="outline" onClick={() => setActiveTab("add")}>
            <Plus className="h-4 w-4 mr-2" />
            Add New Fence
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-4">
            <TabsTrigger value="list">GEO Fence List</TabsTrigger>
            <TabsTrigger value="add">{editingFence ? "Edit" : "Add"} GEO Fence</TabsTrigger>
            <TabsTrigger value="map">Map View</TabsTrigger>
          </TabsList>

          <TabsContent value="list">
            <div className="space-y-4">
              {geoFences.length === 0 ? (
                <div className="text-center py-8">
                  <MapPin className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">No GEO fences defined yet</p>
                  <Button variant="outline" className="mt-4" onClick={() => setActiveTab("add")}>
                    <Plus className="h-4 w-4 mr-2" />
                    Add Your First GEO Fence
                  </Button>
                </div>
              ) : (
                geoFences.map((fence) => (
                  <div key={fence.id} className={`border rounded-lg p-4 ${fence.enabled ? "" : "bg-muted/30"}`}>
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-3">
                        <div className={`p-2 rounded-full ${fence.enabled ? "bg-primary/10" : "bg-muted"}`}>
                          {fence.icon && (
                            <fence.icon
                              className={`h-5 w-5 ${fence.enabled ? "text-primary" : "text-muted-foreground"}`}
                            />
                          )}
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <h3 className="font-medium">{fence.name}</h3>
                            <Badge variant={fence.enabled ? "default" : "outline"}>
                              {fence.enabled ? "Active" : "Disabled"}
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground mt-1">{fence.address}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <Badge variant="outline" className="text-xs">
                              {fence.radius}m radius
                            </Badge>
                            <Badge variant="outline" className="text-xs">
                              {fence.schedule}
                            </Badge>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Switch checked={fence.enabled} onCheckedChange={() => handleToggle(fence.id)} />
                        <Button variant="ghost" size="icon" onClick={() => handleEdit(fence.id)}>
                          <Edit className="h-4 w-4" />
                        </Button>
                        <Dialog>
                          <DialogTrigger asChild>
                            <Button variant="ghost" size="icon">
                              <Trash2 className="h-4 w-4 text-destructive" />
                            </Button>
                          </DialogTrigger>
                          <DialogContent>
                            <DialogHeader>
                              <DialogTitle>Delete GEO Fence</DialogTitle>
                              <DialogDescription>
                                Are you sure you want to delete the "{fence.name}" GEO fence? This action cannot be
                                undone.
                              </DialogDescription>
                            </DialogHeader>
                            <DialogFooter>
                              <Button variant="outline" onClick={() => {}}>
                                Cancel
                              </Button>
                              <Button variant="destructive" onClick={() => handleDelete(fence.id)}>
                                Delete
                              </Button>
                            </DialogFooter>
                          </DialogContent>
                        </Dialog>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </TabsContent>

          <TabsContent value="add">
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Location Name</Label>
                  <Input
                    id="name"
                    placeholder="e.g., School, Library"
                    value={newFence.name}
                    onChange={(e) => setNewFence({ ...newFence, name: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="address">Address</Label>
                  <Input
                    id="address"
                    placeholder="Enter address"
                    value={newFence.address}
                    onChange={(e) => setNewFence({ ...newFence, address: e.target.value })}
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label>Radius (meters)</Label>
                <div className="flex items-center gap-4">
                  <Slider
                    value={[newFence.radius]}
                    min={50}
                    max={500}
                    step={10}
                    onValueChange={(value) => setNewFence({ ...newFence, radius: value[0] })}
                    className="flex-1"
                  />
                  <span className="w-12 text-center">{newFence.radius}m</span>
                </div>
                <p className="text-xs text-muted-foreground mt-1">Define how large the GEO fence area should be</p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="schedule">Schedule</Label>
                <Select
                  value={newFence.schedule}
                  onValueChange={(value) => setNewFence({ ...newFence, schedule: value })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select schedule" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="All days, All hours">All days, All hours</SelectItem>
                    <SelectItem value="Mon-Fri, 8:00 AM - 3:00 PM">Mon-Fri, 8:00 AM - 3:00 PM</SelectItem>
                    <SelectItem value="Mon-Fri, 4:00 PM - 8:00 PM">Mon-Fri, 4:00 PM - 8:00 PM</SelectItem>
                    <SelectItem value="Weekends only">Weekends only</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center space-x-2 mt-4">
                <Switch
                  id="enabled"
                  checked={newFence.enabled}
                  onCheckedChange={(checked) => setNewFence({ ...newFence, enabled: checked })}
                />
                <Label htmlFor="enabled">Enable this GEO fence immediately</Label>
              </div>

              <div className="flex justify-end gap-2 mt-6">
                <Button variant="outline" onClick={handleCancel}>
                  Cancel
                </Button>
                <Button onClick={handleSave} disabled={!newFence.name || !newFence.address}>
                  <Save className="h-4 w-4 mr-2" />
                  {editingFence ? "Update" : "Save"} GEO Fence
                </Button>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="map">
            <div className="border rounded-lg h-[400px] flex items-center justify-center bg-muted/30">
              <div className="text-center p-6">
                <MapPin className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="font-medium mb-2">Interactive Map View</h3>
                <p className="text-sm text-muted-foreground mb-4">View and edit your GEO fences visually on a map</p>
                <Button variant="outline">Load Interactive Map</Button>
              </div>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Note: The map view allows you to visually create and adjust GEO fences by dragging and resizing on the
              map.
            </p>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
