"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { AlertTriangle, Bell, Clock, PhoneCall, Save, Watch } from "lucide-react"

export function DeviceSafetyAlerts() {
  const [wristRemovalAlert, setWristRemovalAlert] = useState(true)
  const [notifyContacts, setNotifyContacts] = useState(true)
  const [alertDelay, setAlertDelay] = useState("immediate")
  const [soundAlert, setSoundAlert] = useState(true)
  const [vibrationAlert, setVibrationAlert] = useState(true)
  const [customMessage, setCustomMessage] = useState("Avani's device has been removed. Please check on them.")
  const [testMode, setTestMode] = useState(false)

  // Sample alert history
  const alertHistory = [
    {
      id: 1,
      type: "removal",
      date: "2025-03-22T14:30:00",
      location: "School",
      duration: "2 minutes",
      resolved: true,
    },
    {
      id: 2,
      type: "removal",
      date: "2025-03-18T09:15:00",
      location: "Home",
      duration: "5 minutes",
      resolved: true,
    },
    {
      id: 3,
      type: "removal",
      date: "2025-03-10T16:45:00",
      location: "Park",
      duration: "8 minutes",
      resolved: true,
    },
  ]

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Watch className="h-5 w-5 text-primary" />
          <CardTitle>Device Removal Alerts</CardTitle>
        </div>
        <CardDescription>Get notified immediately if the device is removed from your child's wrist</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="bg-yellow-50 dark:bg-yellow-950/30 border border-yellow-200 dark:border-yellow-900 p-4 rounded-lg flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-yellow-600 dark:text-yellow-500 mt-0.5" />
          <div>
            <h3 className="font-medium text-sm text-yellow-800 dark:text-yellow-400">Important Safety Feature</h3>
            <p className="text-sm text-yellow-700 dark:text-yellow-500 mt-1">
              Wrist removal detection helps ensure your child's device stays with them at all times. You'll receive an
              immediate alert if the device is removed unexpectedly.
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex items-center justify-between space-x-2">
            <div>
              <Label htmlFor="wrist-removal" className="block">
                Wrist Removal Alerts
              </Label>
              <p className="text-sm text-muted-foreground">Notify when device is removed from wrist</p>
            </div>
            <Switch id="wrist-removal" checked={wristRemovalAlert} onCheckedChange={setWristRemovalAlert} />
          </div>

          <div className="flex items-center justify-between space-x-2">
            <div>
              <Label htmlFor="notify-contacts" className="block">
                Notify Emergency Contacts
              </Label>
              <p className="text-sm text-muted-foreground">Alert primary emergency contacts</p>
            </div>
            <Switch
              id="notify-contacts"
              checked={notifyContacts}
              onCheckedChange={setNotifyContacts}
              disabled={!wristRemovalAlert}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="alert-delay">Alert Delay</Label>
            <Select value={alertDelay} onValueChange={setAlertDelay} disabled={!wristRemovalAlert}>
              <SelectTrigger id="alert-delay">
                <SelectValue placeholder="Select delay" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="immediate">Immediate</SelectItem>
                <SelectItem value="30seconds">After 30 seconds</SelectItem>
                <SelectItem value="1minute">After 1 minute</SelectItem>
                <SelectItem value="2minutes">After 2 minutes</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Set a delay to prevent false alarms during normal activities
            </p>
          </div>

          <div className="flex items-center justify-between space-x-2">
            <div>
              <Label htmlFor="sound-alert" className="block">
                Sound Alert
              </Label>
              <p className="text-sm text-muted-foreground">Play sound on device and app</p>
            </div>
            <Switch
              id="sound-alert"
              checked={soundAlert}
              onCheckedChange={setSoundAlert}
              disabled={!wristRemovalAlert}
            />
          </div>

          <div className="flex items-center justify-between space-x-2">
            <div>
              <Label htmlFor="vibration-alert" className="block">
                Vibration Alert
              </Label>
              <p className="text-sm text-muted-foreground">Vibrate your phone when alert triggers</p>
            </div>
            <Switch
              id="vibration-alert"
              checked={vibrationAlert}
              onCheckedChange={setVibrationAlert}
              disabled={!wristRemovalAlert}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="custom-message">Custom Alert Message</Label>
            <Input
              id="custom-message"
              value={customMessage}
              onChange={(e) => setCustomMessage(e.target.value)}
              disabled={!wristRemovalAlert}
            />
            <p className="text-xs text-muted-foreground">This message will be sent to emergency contacts</p>
          </div>

          <div className="flex items-center justify-between space-x-2">
            <div>
              <Label htmlFor="test-mode" className="block">
                Test Mode
              </Label>
              <p className="text-sm text-muted-foreground">Simulate alerts to test the system</p>
            </div>
            <Switch id="test-mode" checked={testMode} onCheckedChange={setTestMode} disabled={!wristRemovalAlert} />
          </div>

          {testMode && (
            <div className="flex justify-center mt-2">
              <Button variant="outline" className="w-full">
                <Bell className="h-4 w-4 mr-2" />
                Send Test Alert
              </Button>
            </div>
          )}
        </div>

        <div className="flex justify-end">
          <Button>
            <Save className="h-4 w-4 mr-2" />
            Save Settings
          </Button>
        </div>

        <div className="space-y-3">
          <h3 className="font-medium text-sm">Recent Alert History</h3>
          {alertHistory.map((alert) => (
            <div key={alert.id} className="border rounded-lg p-3">
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3">
                  <div className="bg-primary/10 p-2 rounded-full">
                    <Watch className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h4 className="text-sm font-medium">Device Removed</h4>
                      <Badge variant="outline" className="text-xs">
                        {alert.resolved ? "Resolved" : "Active"}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <Clock className="h-3 w-3 text-muted-foreground" />
                      <p className="text-xs text-muted-foreground">
                        {new Date(alert.date).toLocaleDateString()} at{" "}
                        {new Date(alert.date).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                      </p>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      Location: {alert.location} • Duration: {alert.duration}
                    </p>
                  </div>
                </div>
                <Button variant="ghost" size="sm" className="h-7">
                  <PhoneCall className="h-3 w-3 mr-1" />
                  Details
                </Button>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
