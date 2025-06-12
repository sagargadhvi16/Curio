import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { BookOpen, Lightbulb, Microscope, Rocket } from "lucide-react"
import { WeeklySummary } from "@/app/page"

type props = {
  suggested_activities: WeeklySummary['suggested_activities']
}

export function SuggestedActivities({ suggested_activities }: props) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Suggested Activities</CardTitle>
          <CardDescription>Personalized activities based on your child's interests</CardDescription>
        </div>
        <Button variant="outline" size="sm">
          View All
        </Button>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {suggested_activities.map((activity, idx) => (
            <div key={idx} className="flex items-start gap-4 p-3 rounded-lg hover:bg-muted transition-colors">
              <div className="bg-primary/10 p-2 rounded-full">
                <Lightbulb className="h-4 w-4 text-primary" />
              </div>
              <div className="flex-1 space-y-1">
                <h4 className="font-medium">{`Activity ${idx + 1}`}</h4>
                <p className="text-sm text-muted-foreground">{activity}</p>
              </div>
              <Button variant="ghost" size="sm">
                <Lightbulb className="h-4 w-4 mr-2" />
                Try It
              </Button>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
