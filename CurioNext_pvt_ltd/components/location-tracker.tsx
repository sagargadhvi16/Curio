"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import {
  MapPin,
  Plus,
  Minus,
  Home,
  School,
  ShoppingBag,
  Bike,
  AlertTriangle,
  Save,
  History,
  RefreshCw,
  Shield,
  Settings,
  Eye,
  EyeOff,
  ChevronDown,
  Trash2,
} from "lucide-react"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Calendar } from "@/components/ui/calendar"
import { format } from "date-fns"

// Sample location data for Avani
const sampleLocations = [
  {
    id: 1,
    timestamp: new Date(Date.now() - 1000 * 60 * 5), // 5 minutes ago
    latitude: 12.9716,
    longitude: 77.5946,
    accuracy: 15,
    address: "Near National Public School, Bangalore",
    isInSafeZone: true,
    safeZoneName: "School",
  },
  {
    id: 2,
    timestamp: new Date(Date.now() - 1000 * 60 * 20), // 20 minutes ago
    latitude: 12.9718,
    longitude: 77.5942,
    accuracy: 10,
    address: "School Playground, Bangalore",
    isInSafeZone: true,
    safeZoneName: "School",
  },
  {
    id: 3,
    timestamp: new Date(Date.now() - 1000 * 60 * 60), // 1 hour ago
    latitude: 12.972,
    longitude: 77.593,
    accuracy: 8,
    address: "School Entrance, Bangalore",
    isInSafeZone: true,
    safeZoneName: "School",
  },
  {
    id: 4,
    timestamp: new Date(Date.now() - 1000 * 60 * 90), // 1.5 hours ago
    latitude: 12.968,
    longitude: 77.59,
    accuracy: 12,
    address: "Main Road, Bangalore",
    isInSafeZone: false,
    safeZoneName: null,
  },
  {
    id: 5,
    timestamp: new Date(Date.now() - 1000 * 60 * 120), // 2 hours ago
    latitude: 12.965,
    longitude: 77.588,
    accuracy: 5,
    address: "Home, Bangalore",
    isInSafeZone: true,
    safeZoneName: "Home",
  },
]

// Sample safe zones
const sampleSafeZones = [
  {
    id: 1,
    name: "Home",
    address: "123 Residency Road, Bangalore",
    latitude: 12.965,
    longitude: 77.588,
    radius: 100,
    color: "#4CAF50",
    icon: Home,
    isActive: true,
    schedule: "All days, All hours",
    alertsEnabled: true,
  },
  {
    id: 2,
    name: "School",
    address: "National Public School, Bangalore",
    latitude: 12.9716,
    longitude: 77.5946,
    radius: 200,
    color: "#2196F3",
    icon: School,
    isActive: true,
    schedule: "Mon-Fri, 8:00 AM - 4:00 PM",
    alertsEnabled: true,
  },
  {
    id: 3,
    name: "Grandparents' House",
    address: "456 MG Road, Bangalore",
    latitude: 12.975,
    longitude: 77.6,
    radius: 150,
    color: "#9C27B0",
    icon: Home,
    isActive: true,
    schedule: "Weekends, All hours",
    alertsEnabled: true,
  },
  {
    id: 4,
    name: "Shopping Mall",
    address: "Forum Mall, Bangalore",
    latitude: 12.934,
    longitude: 77.6155,
    radius: 300,
    color: "#FF9800",
    icon: ShoppingBag,
    isActive: false,
    schedule: "All days, 10:00 AM - 8:00 PM",
    alertsEnabled: true,
  },
]

// Sample alerts
const sampleAlerts = [
  {
    id: 1,
    timestamp: new Date(Date.now() - 1000 * 60 * 30), // 30 minutes ago
    type: "exit",
    safeZoneName: "School",
    location: {
      latitude: 12.973,
      longitude: 77.596,
      address: "100m from School Boundary, Bangalore",
    },
    resolved: true,
    resolvedAt: new Date(Date.now() - 1000 * 60 * 25), // 25 minutes ago
  },
  {
    id: 2,
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 3), // 3 hours ago
    type: "exit",
    safeZoneName: "Home",
    location: {
      latitude: 12.966,
      longitude: 77.59,
      address: "Near Home, Bangalore",
    },
    resolved: true,
    resolvedAt: new Date(Date.now() - 1000 * 60 * 60 * 2.9), // 2.9 hours ago
  },
  {
    id: 3,
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24), // 1 day ago
    type: "exit",
    safeZoneName: "School",
    location: {
      latitude: 12.974,
      longitude: 77.597,
      address: "School Exit, Bangalore",
    },
    resolved: true,
    resolvedAt: new Date(Date.now() - 1000 * 60 * 60 * 23.9), // 23.9 hours ago
  },
  {
    id: 4,
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 48), // 2 days ago
    type: "battery",
    safeZoneName: null,
    location: {
      latitude: 12.965,
      longitude: 77.588,
      address: "Home, Bangalore",
    },
    resolved: true,
    resolvedAt: new Date(Date.now() - 1000 * 60 * 60 * 47), // 47 hours ago
  },
]

export function LocationTracker() {
  const [activeTab, setActiveTab] = useState("map")
  const [currentLocation, setCurrentLocation] = useState(sampleLocations[0])
  const [safeZones, setSafeZones] = useState(sampleSafeZones)
  const [locationHistory, setLocationHistory] = useState(sampleLocations)
  const [alerts, setAlerts] = useState(sampleAlerts)
  const [isTracking, setIsTracking] = useState(true)
  const [showAllSafeZones, setShowAllSafeZones] = useState(true)
  const [selectedSafeZone, setSelectedSafeZone] = useState<number | null>(null)
  const [isAddingSafeZone, setIsAddingSafeZone] = useState(false)
  const [newSafeZone, setNewSafeZone] = useState({
    name: "",
    address: "",
    radius: 100,
    schedule: "All days, All hours",
    alertsEnabled: true,
  })
  const [mapZoom, setMapZoom] = useState(15)
  const [date, setDate] = useState<Date | undefined>(new Date())
  const [showPrivacySettings, setShowPrivacySettings] = useState(false)
  const [privacySettings, setPrivacySettings] = useState({
    shareLocationWithParents: true,
    shareLocationWithEmergencyContacts: true,
    recordLocationHistory: true,
    locationHistoryRetention: "30days",
    showAccuracyRadius: true,
  })
  const [isLoading, setIsLoading] = useState(false)
  const mapRef = useRef<HTMLDivElement>(null)

  // Simulate real-time location updates
  useEffect(() => {
    if (!isTracking) return

    const interval = setInterval(() => {
      // Simulate small movement
      const newLat = currentLocation.latitude + (Math.random() - 0.5) * 0.001
      const newLng = currentLocation.longitude + (Math.random() - 0.5) * 0.001

      // Check if in any safe zone
      const inSafeZone = safeZones.find((zone) => {
        if (!zone.isActive) return false

        // Calculate distance between points using Haversine formula
        const R = 6371e3 // Earth radius in meters
        const φ1 = (zone.latitude * Math.PI) / 180
        const φ2 = (newLat * Math.PI) / 180
        const Δφ = ((newLat - zone.latitude) * Math.PI) / 180
        const Δλ = ((newLng - zone.longitude) * Math.PI) / 180

        const a =
          Math.sin(Δφ / 2) * Math.sin(Δφ / 2) + Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) * Math.sin(Δλ / 2)
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
        const distance = R * c

        return distance <= zone.radius
      })

      const newLocation = {
        id: currentLocation.id + 1,
        timestamp: new Date(),
        latitude: newLat,
        longitude: newLng,
        accuracy: Math.floor(Math.random() * 10) + 5, // Random accuracy between 5-15m
        address: currentLocation.address, // In a real app, this would be reverse geocoded
        isInSafeZone: !!inSafeZone,
        safeZoneName: inSafeZone ? inSafeZone.name : null,
      }

      setCurrentLocation(newLocation)
      setLocationHistory((prev) => [newLocation, ...prev])

      // Check if we need to generate an alert
      if (currentLocation.isInSafeZone && !newLocation.isInSafeZone) {
        const newAlert = {
          id: alerts.length + 1,
          timestamp: new Date(),
          type: "exit",
          safeZoneName: currentLocation.safeZoneName,
          location: {
            latitude: newLat,
            longitude: newLng,
            address: currentLocation.address,
          },
          resolved: false,
          resolvedAt: null,
        }
        setAlerts((prev) => [newAlert, ...prev])
      }
    }, 10000) // Update every 10 seconds

    return () => clearInterval(interval)
  }, [currentLocation, isTracking, safeZones])

  const handleZoomIn = () => {
    setMapZoom((prev) => Math.min(prev + 1, 18))
  }

  const handleZoomOut = () => {
    setMapZoom((prev) => Math.max(prev - 1, 10))
  }

  const handleToggleSafeZone = (id: number) => {
    setSafeZones((prev) => prev.map((zone) => (zone.id === id ? { ...zone, isActive: !zone.isActive } : zone)))
  }

  const handleSaveSafeZone = () => {
    if (!newSafeZone.name || !newSafeZone.address) return

    // In a real app, we would geocode the address to get lat/lng
    const newZone = {
      id: safeZones.length + 1,
      name: newSafeZone.name,
      address: newSafeZone.address,
      latitude: currentLocation.latitude,
      longitude: currentLocation.longitude,
      radius: newSafeZone.radius,
      color: ["#4CAF50", "#2196F3", "#9C27B0", "#FF9800", "#F44336"][Math.floor(Math.random() * 5)],
      icon: [Home, School, ShoppingBag, Bike][Math.floor(Math.random() * 4)],
      isActive: true,
      schedule: newSafeZone.schedule,
      alertsEnabled: newSafeZone.alertsEnabled,
    }

    setSafeZones((prev) => [...prev, newZone])
    setIsAddingSafeZone(false)
    setNewSafeZone({
      name: "",
      address: "",
      radius: 100,
      schedule: "All days, All hours",
      alertsEnabled: true,
    })
  }

  const handleRefreshLocation = () => {
    setIsLoading(true)

    // Simulate loading
    setTimeout(() => {
      // Simulate a small movement
      const newLat = currentLocation.latitude + (Math.random() - 0.5) * 0.0005
      const newLng = currentLocation.longitude + (Math.random() - 0.5) * 0.0005

      const newLocation = {
        ...currentLocation,
        timestamp: new Date(),
        latitude: newLat,
        longitude: newLng,
        accuracy: Math.floor(Math.random() * 10) + 5,
      }

      setCurrentLocation(newLocation)
      setIsLoading(false)
    }, 1500)
  }

  const handleFilterHistory = (date: Date | undefined) => {
    setDate(date)
    // In a real app, we would filter the location history based on the date
  }

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  }

  const formatDate = (date: Date) => {
    const now = new Date()
    const yesterday = new Date(now)
    yesterday.setDate(yesterday.getDate() - 1)

    if (date.toDateString() === now.toDateString()) {
      return `Today, ${formatTime(date)}`
    } else if (date.toDateString() === yesterday.toDateString()) {
      return `Yesterday, ${formatTime(date)}`
    } else {
      return `${date.toLocaleDateString()} ${formatTime(date)}`
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Avani's Location Tracker</CardTitle>
            <CardDescription>Monitor your child's location in real-time and set up safe zones</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center space-x-2">
              <Switch id="tracking-toggle" checked={isTracking} onCheckedChange={setIsTracking} />
              <Label htmlFor="tracking-toggle">{isTracking ? "Tracking Active" : "Tracking Paused"}</Label>
            </div>
            <Button variant="outline" size="icon" onClick={() => setShowPrivacySettings(true)} title="Privacy Settings">
              <Shield className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="mb-4">
              <TabsTrigger value="map">Live Map</TabsTrigger>
              <TabsTrigger value="safezones">Safe Zones</TabsTrigger>
              <TabsTrigger value="history">Location History</TabsTrigger>
              <TabsTrigger value="alerts">Alerts</TabsTrigger>
            </TabsList>

            {/* Live Map Tab */}
            <TabsContent value="map">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Avatar className="h-10 w-10 border-2 border-primary">
                      <AvatarImage src="/placeholder.svg?height=40&width=40" alt="Avani" />
                      <AvatarFallback>AV</AvatarFallback>
                    </Avatar>
                    <div>
                      <p className="font-medium">Avani</p>
                      <p className="text-sm text-muted-foreground">
                        Last updated: {formatTime(currentLocation.timestamp)}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={currentLocation.isInSafeZone ? "default" : "destructive"}>
                      {currentLocation.isInSafeZone
                        ? `In ${currentLocation.safeZoneName} Safe Zone`
                        : "Outside Safe Zone"}
                    </Badge>
                    <Button variant="outline" size="sm" onClick={handleRefreshLocation} disabled={isLoading}>
                      {isLoading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                      <span className="ml-2">Refresh</span>
                    </Button>
                  </div>
                </div>

                {/* Map Container */}
                <div className="relative h-[500px] rounded-lg border bg-muted overflow-hidden" ref={mapRef}>
                  {/* This would be replaced with an actual map library like Leaflet or Google Maps */}
                  <div className="absolute inset-0 bg-gray-100 dark:bg-gray-800">
                    {/* Simulated map with CSS */}
                    <div className="w-full h-full relative overflow-hidden">
                      {/* Map background */}
                      <div className="absolute inset-0 bg-[url('/placeholder.svg?height=1000&width=1000')] bg-cover opacity-50"></div>

                      {/* Safe zones */}
                      {showAllSafeZones &&
                        safeZones
                          .filter((zone) => zone.isActive)
                          .map((zone) => (
                            <div
                              key={zone.id}
                              className="absolute rounded-full border-2 border-dashed transform -translate-x-1/2 -translate-y-1/2"
                              style={{
                                left: `${50 + (zone.longitude - currentLocation.longitude) * 10000}%`,
                                top: `${50 - (zone.latitude - currentLocation.latitude) * 10000}%`,
                                width: `${zone.radius / 5}px`,
                                height: `${zone.radius / 5}px`,
                                backgroundColor: `${zone.color}20`,
                                borderColor: zone.color,
                              }}
                            >
                              <div
                                className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 p-1 rounded-full"
                                style={{ backgroundColor: zone.color }}
                              >
                                <zone.icon className="h-4 w-4 text-white" />
                              </div>
                            </div>
                          ))}

                      {/* Current location marker */}
                      <div
                        className="absolute w-6 h-6 bg-primary rounded-full border-2 border-white shadow-lg transform -translate-x-1/2 -translate-y-1/2 flex items-center justify-center"
                        style={{
                          left: "50%",
                          top: "50%",
                          zIndex: 1000,
                        }}
                      >
                        <MapPin className="h-4 w-4 text-white" />
                      </div>

                      {/* Accuracy circle */}
                      {privacySettings.showAccuracyRadius && (
                        <div
                          className="absolute rounded-full border border-primary/50 bg-primary/20 transform -translate-x-1/2 -translate-y-1/2"
                          style={{
                            left: "50%",
                            top: "50%",
                            width: `${currentLocation.accuracy * 2}px`,
                            height: `${currentLocation.accuracy * 2}px`,
                          }}
                        />
                      )}
                    </div>
                  </div>

                  {/* Map Controls */}
                  <div className="absolute top-4 right-4 flex flex-col gap-2">
                    <Button variant="secondary" size="icon" onClick={handleZoomIn}>
                      <Plus className="h-4 w-4" />
                    </Button>
                    <Button variant="secondary" size="icon" onClick={handleZoomOut}>
                      <Minus className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="secondary"
                      size="icon"
                      onClick={() => setShowAllSafeZones(!showAllSafeZones)}
                      title={showAllSafeZones ? "Hide Safe Zones" : "Show Safe Zones"}
                    >
                      {showAllSafeZones ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                  </div>

                  {/* Location Info */}
                  <div className="absolute bottom-4 left-4 right-4 bg-card/90 backdrop-blur-sm p-3 rounded-lg border shadow-sm">
                    <div className="flex flex-col gap-1">
                      <div className="flex items-center justify-between">
                        <p className="font-medium">Current Location</p>
                        <Badge variant="outline" className="text-xs">
                          Accuracy: ±{currentLocation.accuracy}m
                        </Badge>
                      </div>
                      <p className="text-sm">{currentLocation.address}</p>
                      <p className="text-xs text-muted-foreground">
                        Coordinates: {currentLocation.latitude.toFixed(6)}, {currentLocation.longitude.toFixed(6)}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* Safe Zones Tab */}
            <TabsContent value="safezones">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-muted-foreground">
                    Define areas where your child is allowed to be. You'll receive alerts if they leave these zones.
                  </p>
                  <Button onClick={() => setIsAddingSafeZone(true)}>
                    <Plus className="h-4 w-4 mr-2" />
                    Add Safe Zone
                  </Button>
                </div>

                {isAddingSafeZone && (
                  <Card className="border-dashed">
                    <CardHeader>
                      <CardTitle>Add New Safe Zone</CardTitle>
                      <CardDescription>Define a new safe area for your child</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <Label htmlFor="zone-name">Zone Name</Label>
                            <Input
                              id="zone-name"
                              placeholder="e.g., School, Home, Grandparents"
                              value={newSafeZone.name}
                              onChange={(e) => setNewSafeZone({ ...newSafeZone, name: e.target.value })}
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="zone-address">Address</Label>
                            <Input
                              id="zone-address"
                              placeholder="Enter address"
                              value={newSafeZone.address}
                              onChange={(e) => setNewSafeZone({ ...newSafeZone, address: e.target.value })}
                            />
                          </div>
                        </div>

                        <div className="space-y-2">
                          <Label>Safe Zone Radius: {newSafeZone.radius}m</Label>
                          <Slider
                            value={[newSafeZone.radius]}
                            min={50}
                            max={500}
                            step={10}
                            onValueChange={(value) => setNewSafeZone({ ...newSafeZone, radius: value[0] })}
                          />
                          <p className="text-xs text-muted-foreground">
                            Define how large the safe zone should be around this location
                          </p>
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="zone-schedule">Schedule</Label>
                          <Select
                            value={newSafeZone.schedule}
                            onValueChange={(value) => setNewSafeZone({ ...newSafeZone, schedule: value })}
                          >
                            <SelectTrigger id="zone-schedule">
                              <SelectValue placeholder="Select schedule" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="All days, All hours">All days, All hours</SelectItem>
                              <SelectItem value="Mon-Fri, 8:00 AM - 4:00 PM">Mon-Fri, 8:00 AM - 4:00 PM</SelectItem>
                              <SelectItem value="Weekends, All hours">Weekends, All hours</SelectItem>
                              <SelectItem value="Custom">Custom</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="flex items-center space-x-2">
                          <Switch
                            id="zone-alerts"
                            checked={newSafeZone.alertsEnabled}
                            onCheckedChange={(checked) => setNewSafeZone({ ...newSafeZone, alertsEnabled: checked })}
                          />
                          <Label htmlFor="zone-alerts">Enable alerts for this zone</Label>
                        </div>
                      </div>
                    </CardContent>
                    <CardFooter className="flex justify-end gap-2">
                      <Button variant="outline" onClick={() => setIsAddingSafeZone(false)}>
                        Cancel
                      </Button>
                      <Button onClick={handleSaveSafeZone} disabled={!newSafeZone.name || !newSafeZone.address}>
                        <Save className="h-4 w-4 mr-2" />
                        Save Safe Zone
                      </Button>
                    </CardFooter>
                  </Card>
                )}

                <div className="space-y-3">
                  {safeZones.map((zone) => (
                    <div
                      key={zone.id}
                      className={`border rounded-lg p-4 transition-colors ${zone.isActive ? "" : "bg-muted/30"}`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          <div
                            className={`p-2 rounded-full ${zone.isActive ? "bg-primary/10" : "bg-muted"}`}
                            style={{ backgroundColor: zone.isActive ? `${zone.color}20` : undefined }}
                          >
                            <zone.icon
                              className={`h-5 w-5 ${zone.isActive ? "text-primary" : "text-muted-foreground"}`}
                            />
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                              <h3 className="font-medium">{zone.name}</h3>
                              <Badge variant={zone.isActive ? "default" : "outline"}>
                                {zone.isActive ? "Active" : "Inactive"}
                              </Badge>
                            </div>
                            <p className="text-sm text-muted-foreground mt-1">{zone.address}</p>
                            <div className="flex items-center gap-2 mt-1">
                              <Badge variant="outline" className="text-xs">
                                {zone.radius}m radius
                              </Badge>
                              <Badge variant="outline" className="text-xs">
                                {zone.schedule}
                              </Badge>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Switch checked={zone.isActive} onCheckedChange={() => handleToggleSafeZone(zone.id)} />
                          <Button variant="ghost" size="sm" onClick={() => setSelectedSafeZone(zone.id)}>
                            <Settings className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </TabsContent>

            {/* Location History Tab */}
            <TabsContent value="history">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-muted-foreground">View your child's location history over time</p>
                  <div className="flex items-center gap-2">
                    <Popover>
                      <PopoverTrigger asChild>
                        <Button variant="outline" className="flex items-center gap-2">
                          {date ? format(date, "PPP") : "Pick a date"}
                          <ChevronDown className="h-4 w-4" />
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-auto p-0">
                        <Calendar mode="single" selected={date} onSelect={handleFilterHistory} initialFocus />
                      </PopoverContent>
                    </Popover>
                    <Button variant="outline" onClick={() => setDate(new Date())}>
                      Today
                    </Button>
                  </div>
                </div>

                <div className="space-y-3">
                  {locationHistory.map((location) => (
                    <div key={location.id} className="border rounded-lg p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          <div
                            className={`p-2 rounded-full ${location.isInSafeZone ? "bg-primary/10" : "bg-destructive/10"}`}
                          >
                            <MapPin
                              className={`h-5 w-5 ${location.isInSafeZone ? "text-primary" : "text-destructive"}`}
                            />
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                              <h3 className="font-medium">{location.address}</h3>
                              {location.isInSafeZone ? (
                                <Badge variant="default">{location.safeZoneName}</Badge>
                              ) : (
                                <Badge variant="destructive">Outside Safe Zone</Badge>
                              )}
                            </div>
                            <p className="text-sm text-muted-foreground mt-1">{formatDate(location.timestamp)}</p>
                            <p className="text-xs text-muted-foreground mt-1">
                              Coordinates: {location.latitude.toFixed(6)}, {location.longitude.toFixed(6)} • Accuracy: ±
                              {location.accuracy}m
                            </p>
                          </div>
                        </div>
                        <Button variant="ghost" size="sm">
                          <MapPin className="h-4 w-4 mr-2" />
                          View on Map
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </TabsContent>

            {/* Alerts Tab */}
            <TabsContent value="alerts">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-muted-foreground">View alerts when your child leaves a safe zone</p>
                  <Button variant="outline">
                    <History className="h-4 w-4 mr-2" />
                    View All Alerts
                  </Button>
                </div>

                {alerts.length === 0 ? (
                  <div className="text-center py-12">
                    <AlertTriangle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground">No alerts to display</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {alerts.map((alert) => (
                      <div
                        key={alert.id}
                        className={`border rounded-lg p-4 ${alert.resolved ? "" : "border-destructive bg-destructive/5"}`}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex items-start gap-3">
                            <div className={`p-2 rounded-full ${alert.resolved ? "bg-muted" : "bg-destructive/10"}`}>
                              <AlertTriangle
                                className={`h-5 w-5 ${alert.resolved ? "text-muted-foreground" : "text-destructive"}`}
                              />
                            </div>
                            <div>
                              <div className="flex items-center gap-2">
                                <h3 className="font-medium">
                                  {alert.type === "exit" ? `Left ${alert.safeZoneName} Safe Zone` : "Low Battery Alert"}
                                </h3>
                                <Badge variant={alert.resolved ? "outline" : "destructive"}>
                                  {alert.resolved ? "Resolved" : "Active"}
                                </Badge>
                              </div>
                              <p className="text-sm text-muted-foreground mt-1">{formatDate(alert.timestamp)}</p>
                              <p className="text-xs text-muted-foreground mt-1">Location: {alert.location.address}</p>
                              {alert.resolved && (
                                <p className="text-xs text-muted-foreground mt-1">
                                  Resolved at: {formatDate(alert.resolvedAt!)}
                                </p>
                              )}
                            </div>
                          </div>
                          <Button variant="ghost" size="sm">
                            <MapPin className="h-4 w-4 mr-2" />
                            View on Map
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Privacy Settings Dialog */}
      <Dialog open={showPrivacySettings} onOpenChange={setShowPrivacySettings}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Location Privacy Settings</DialogTitle>
            <DialogDescription>Configure how location data is collected, stored, and shared</DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="flex items-center justify-between space-x-2">
              <div>
                <Label htmlFor="share-parents" className="block">
                  Share Location with Parents
                </Label>
                <p className="text-sm text-muted-foreground">Allow parents to view real-time location</p>
              </div>
              <Switch
                id="share-parents"
                checked={privacySettings.shareLocationWithParents}
                onCheckedChange={(checked) =>
                  setPrivacySettings({ ...privacySettings, shareLocationWithParents: checked })
                }
              />
            </div>

            <div className="flex items-center justify-between space-x-2">
              <div>
                <Label htmlFor="share-emergency" className="block">
                  Share with Emergency Contacts
                </Label>
                <p className="text-sm text-muted-foreground">Allow emergency contacts to view location during alerts</p>
              </div>
              <Switch
                id="share-emergency"
                checked={privacySettings.shareLocationWithEmergencyContacts}
                onCheckedChange={(checked) =>
                  setPrivacySettings({ ...privacySettings, shareLocationWithEmergencyContacts: checked })
                }
              />
            </div>

            <div className="flex items-center justify-between space-x-2">
              <div>
                <Label htmlFor="record-history" className="block">
                  Record Location History
                </Label>
                <p className="text-sm text-muted-foreground">Store location data for historical viewing</p>
              </div>
              <Switch
                id="record-history"
                checked={privacySettings.recordLocationHistory}
                onCheckedChange={(checked) =>
                  setPrivacySettings({ ...privacySettings, recordLocationHistory: checked })
                }
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="history-retention">Location History Retention</Label>
              <Select
                value={privacySettings.locationHistoryRetention}
                onValueChange={(value) => setPrivacySettings({ ...privacySettings, locationHistoryRetention: value })}
                disabled={!privacySettings.recordLocationHistory}
              >
                <SelectTrigger id="history-retention">
                  <SelectValue placeholder="Select retention period" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="7days">7 days</SelectItem>
                  <SelectItem value="30days">30 days</SelectItem>
                  <SelectItem value="90days">90 days</SelectItem>
                  <SelectItem value="1year">1 year</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center justify-between space-x-2">
              <div>
                <Label htmlFor="show-accuracy" className="block">
                  Show Accuracy Radius
                </Label>
                <p className="text-sm text-muted-foreground">Display location accuracy circle on map</p>
              </div>
              <Switch
                id="show-accuracy"
                checked={privacySettings.showAccuracyRadius}
                onCheckedChange={(checked) => setPrivacySettings({ ...privacySettings, showAccuracyRadius: checked })}
              />
            </div>

            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Privacy Notice</AlertTitle>
              <AlertDescription>
                Location data is encrypted and stored securely. You can delete all location history at any time.
              </AlertDescription>
            </Alert>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowPrivacySettings(false)}>
              Cancel
            </Button>
            <Button onClick={() => setShowPrivacySettings(false)}>Save Settings</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Safe Zone Settings Dialog */}
      <Dialog open={selectedSafeZone !== null} onOpenChange={(open) => !open && setSelectedSafeZone(null)}>
        <DialogContent className="sm:max-w-[500px]">
          {selectedSafeZone !== null && (
            <>
              <DialogHeader>
                <DialogTitle>Edit Safe Zone</DialogTitle>
                <DialogDescription>
                  Modify settings for {safeZones.find((z) => z.id === selectedSafeZone)?.name}
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="edit-zone-name">Zone Name</Label>
                  <Input id="edit-zone-name" defaultValue={safeZones.find((z) => z.id === selectedSafeZone)?.name} />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="edit-zone-address">Address</Label>
                  <Input
                    id="edit-zone-address"
                    defaultValue={safeZones.find((z) => z.id === selectedSafeZone)?.address}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Safe Zone Radius: {safeZones.find((z) => z.id === selectedSafeZone)?.radius}m</Label>
                  <Slider
                    defaultValue={[safeZones.find((z) => z.id === selectedSafeZone)?.radius || 100]}
                    min={50}
                    max={500}
                    step={10}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="edit-zone-schedule">Schedule</Label>
                  <Select defaultValue={safeZones.find((z) => z.id === selectedSafeZone)?.schedule}>
                    <SelectTrigger id="edit-zone-schedule">
                      <SelectValue placeholder="Select schedule" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="All days, All hours">All days, All hours</SelectItem>
                      <SelectItem value="Mon-Fri, 8:00 AM - 4:00 PM">Mon-Fri, 8:00 AM - 4:00 PM</SelectItem>
                      <SelectItem value="Weekends, All hours">Weekends, All hours</SelectItem>
                      <SelectItem value="Custom">Custom</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between space-x-2">
                  <div>
                    <Label htmlFor="edit-zone-alerts" className="block">
                      Enable Alerts
                    </Label>
                    <p className="text-sm text-muted-foreground">Get notified when your child leaves this zone</p>
                  </div>
                  <Switch
                    id="edit-zone-alerts"
                    defaultChecked={safeZones.find((z) => z.id === selectedSafeZone)?.alertsEnabled}
                  />
                </div>

                <div className="pt-4 border-t">
                  <Button variant="destructive" className="w-full">
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete Safe Zone
                  </Button>
                </div>
              </div>

              <DialogFooter>
                <Button variant="outline" onClick={() => setSelectedSafeZone(null)}>
                  Cancel
                </Button>
                <Button onClick={() => setSelectedSafeZone(null)}>Save Changes</Button>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
