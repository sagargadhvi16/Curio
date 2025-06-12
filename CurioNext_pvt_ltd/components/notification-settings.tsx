import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

export function NotificationSettings() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Notification Settings</CardTitle>
        <CardDescription>
          Customize how and when you receive updates about your child's learning journey
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-4">
          <h3 className="text-lg font-medium">Notification Types</h3>

          <div className="space-y-4">
            <div className="flex items-center justify-between space-x-2">
              <div>
                <Label htmlFor="new-interests" className="block">
                  New Interests
                </Label>
                <p className="text-sm text-muted-foreground">Get notified when your child explores new topics</p>
              </div>
              <Switch id="new-interests" defaultChecked />
            </div>

            <div className="flex items-center justify-between space-x-2">
              <div>
                <Label htmlFor="learning-milestones" className="block">
                  Learning Milestones
                </Label>
                <p className="text-sm text-muted-foreground">Receive updates on significant learning achievements</p>
              </div>
              <Switch id="learning-milestones" defaultChecked />
            </div>

            <div className="flex items-center justify-between space-x-2">
              <div>
                <Label htmlFor="weekly-summaries" className="block">
                  Weekly Summaries
                </Label>
                <p className="text-sm text-muted-foreground">Get a weekly overview of your child's learning journey</p>
              </div>
              <Switch id="weekly-summaries" defaultChecked />
            </div>

            <div className="flex items-center justify-between space-x-2">
              <div>
                <Label htmlFor="safety-alerts" className="block">
                  Safety Alerts
                </Label>
                <p className="text-sm text-muted-foreground">Receive immediate notifications for safety concerns</p>
              </div>
              <Switch id="safety-alerts" defaultChecked />
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-lg font-medium">Delivery Method</h3>

          <RadioGroup defaultValue="push-and-email">
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="push-only" id="push-only" />
              <Label htmlFor="push-only">Push notifications only</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="email-only" id="email-only" />
              <Label htmlFor="email-only">Email only</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="push-and-email" id="push-and-email" />
              <Label htmlFor="push-and-email">Both push notifications and email</Label>
            </div>
          </RadioGroup>
        </div>

        <div className="space-y-4">
          <h3 className="text-lg font-medium">Frequency</h3>

          <div className="space-y-2">
            <Label htmlFor="summary-frequency">Weekly Summary Frequency</Label>
            <Select defaultValue="weekly">
              <SelectTrigger id="summary-frequency">
                <SelectValue placeholder="Select frequency" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="daily">Daily</SelectItem>
                <SelectItem value="weekly">Weekly</SelectItem>
                <SelectItem value="biweekly">Bi-weekly</SelectItem>
                <SelectItem value="monthly">Monthly</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="quiet-hours">Quiet Hours</Label>
            <div className="grid grid-cols-2 gap-4">
              <Select defaultValue="20:00">
                <SelectTrigger id="quiet-hours-start">
                  <SelectValue placeholder="Start time" />
                </SelectTrigger>
                <SelectContent>
                  {Array.from({ length: 24 }, (_, i) => i).map((hour) => (
                    <SelectItem key={hour} value={`${hour.toString().padStart(2, "0")}:00`}>
                      {hour.toString().padStart(2, "0")}:00
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select defaultValue="08:00">
                <SelectTrigger id="quiet-hours-end">
                  <SelectValue placeholder="End time" />
                </SelectTrigger>
                <SelectContent>
                  {Array.from({ length: 24 }, (_, i) => i).map((hour) => (
                    <SelectItem key={hour} value={`${hour.toString().padStart(2, "0")}:00`}>
                      {hour.toString().padStart(2, "0")}:00
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
