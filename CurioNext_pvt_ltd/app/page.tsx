"use client"
import { DashboardHeader } from "@/components/dashboard-header"
import { InterestMap } from "@/components/interest-map"
import { Button } from "@/components/ui/button"
import { WeeklySummary } from "@/components/weekly-summary"
import { ExternalLink } from "lucide-react"
import Link from "next/link"
import { useEffect, useState } from "react"
import { SuggestedActivities } from "@/components/suggested-activities"


export interface WeeklySummary {
    interest_distribution: {
        [key: string] : number
    };
    topics: [];
    weekly_summary: string;
    conversation_starters: string[];
    suggested_activities: string[];
    chart_path: string;
}


const getAnalyticsData = async (): Promise<{ data: WeeklySummary | null, error: string | null }> => {
  try {
    const response = await fetch('http://127.0.0.1:8000/assistant/chat_analysis')
    // const response = await fetch('https://nandini-report.free.beeceptor.com/analytics')
    if (response.ok) {
      return {
        data: await response.json(),
        error: null
      }
    } else {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
    return ({
      data: null,
      error: errorMessage
    });
  }
}


export default function Home() {
  const [data, setData] = useState<WeeklySummary | null>( null)
  const [loading, setLoading] = useState(false); // State to track loading


  const fetchData = async () => {
     setLoading(true); // Start loading
    const { data, error} = await getAnalyticsData()
    if(data !== null ) {
      setData(data)
    } else {
      console.error(error)
    }
    setLoading(false)
  }
  const handleRefresh =() => {
    fetchData()
  }
  useEffect(() => {
    fetchData()
  }, [])
  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Overlay */}
      {loading && (
        <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-10">
          <div className="spinner-border animate-spin inline-block w-8 h-8 border-4 rounded-full border-t-transparent border-white"></div>
        </div>
      )}
      
      <DashboardHeader  onRefresh={handleRefresh} />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {data !== null ? <InterestMap  data={data.interest_distribution} /> : null}
        {data !== null ? <WeeklySummary weekly_summary={data.weekly_summary} conversation_starters={data.conversation_starters} /> : null }
        {/* {data !== null ? <SuggestedActivities suggested_activities={data.suggested_activities} /> : null } */}
      </div>

      <div className="flex justify-center mt-4">
        <Button size="lg" className="gap-2" asChild>
          <Link href="https://v0-curiosity-dashboard-elzgiy.vercel.app" target="_blank" rel="noopener noreferrer">
            <ExternalLink className="h-4 w-4" />
            View Full Dashboard
          </Link>
        </Button>
      </div>
    </div>
  )
}
