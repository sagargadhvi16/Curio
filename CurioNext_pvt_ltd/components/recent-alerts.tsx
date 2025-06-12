import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Bell, ChevronRight } from "lucide-react"

// Sample alert data
const alerts = [
  {
    id: 1,
    title: "New Interest Detected",
    description: "Avani has been asking questions about space and astronomy",
    time: "2 hours ago",
    type: "interest",
  },
  {
    id: 2,
    title: "Learning Milestone",
    description: "Avani has explored 10 different topics in 'How Things Work'",
    time: "Yesterday",
    type: "milestone",
  },
  {
    id: 3,
    title: "Weekly Summary Available",
    description: "Listen to Avani's learning journey from the past week",
    time: "2 days ago",
    type: "summary",
  },
]

export function RecentAlerts() {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Recent Alerts</CardTitle>
          <CardDescription>Stay updated on your child's learning milestones</CardDescription>
        </div>
        <Button variant="outline" size="sm">
          View All
        </Button>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {alerts.map((alert) => (
            <div key={alert.id} className="flex items-start gap-4 p-3 rounded-lg hover:bg-muted transition-colors">
              <div className="bg-primary/10 p-2 rounded-full">
                <Bell className="h-4 w-4 text-primary" />
              </div>
              <div className="flex-1 space-y-1">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium">{alert.title}</h4>
                  <Badge
                    variant={
                      alert.type === "interest" ? "default" : alert.type === "milestone" ? "secondary" : "outline"
                    }
                  >
                    {alert.type}
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground">{alert.description}</p>
                <p className="text-xs text-muted-foreground">{alert.time}</p>
              </div>
              <Button variant="ghost" size="icon" className="mt-1">
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
