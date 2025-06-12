"use client"

import { WeeklySummary as IWeeklySummary } from "@/app/page";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

// type Props = Pick<IWeeklySummary, 'weekly_summary' | 'suggested_activities'>;
type Props = Pick<IWeeklySummary, 'weekly_summary' | 'conversation_starters'>;

// export function WeeklySummary( {weekly_summary, suggested_activities }: Props) {
export function WeeklySummary( {weekly_summary, conversation_starters }: Props) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Weekly Summary</CardTitle>
        <CardDescription>A personalized summary of your child's learning journey this week</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="bg-muted p-4 rounded-lg">
          <p className="text-sm">
           {weekly_summary}
          </p>
          {/* <div className="text-xs text-muted-foreground mt-4 text-right">March 18, 2025</div> */}
        </div>

        <div className="space-y-2">
          <h3 className="font-medium">Suggested Conversation Starters</h3>
          <ul className="space-y-2 text-sm">
              {conversation_starters.map((activity, idx) => {
              // {suggested_activities.map((activity, idx) => { 
                return (<li key={idx} className="flex items-start gap-2">
                  <span className="bg-primary/10 text-primary rounded-full w-5 h-5 flex items-center justify-center flex-shrink-0 mt-0.5">
                1
              </span>
              <span>{activity}</span>
            </li>)
              })}

          </ul>
        </div>
      </CardContent>
    </Card>
  )
}
