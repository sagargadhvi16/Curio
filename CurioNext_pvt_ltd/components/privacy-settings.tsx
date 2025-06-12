import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Download, Info } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

export function PrivacySettings() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Privacy Settings</CardTitle>
        <CardDescription>Information about CurioNext's privacy policy and data handling</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <Alert>
          <Info className="h-4 w-4" />
          <AlertTitle>Standardized Privacy Policy</AlertTitle>
          <AlertDescription>
            CurioNext follows a standardized privacy policy that cannot be modified by individual users. This ensures
            consistent protection of children's data across all users of the platform.
          </AlertDescription>
        </Alert>

        <div className="space-y-4">
          <h3 className="text-lg font-medium">What You Can Access</h3>

          <div className="space-y-2 text-sm text-muted-foreground">
            <p>As a parent, you have access to:</p>
            <ul className="list-disc pl-5 space-y-1">
              <li>Interest categories and their relative distribution</li>
              <li>Learning patterns and trends over time</li>
              <li>AI-generated insights and recommendations</li>
              <li>Activity levels and engagement metrics</li>
            </ul>
            <p className="mt-2">You do not have access to:</p>
            <ul className="list-disc pl-5 space-y-1">
              <li>Specific questions asked by your child</li>
              <li>Raw conversation data</li>
              <li>Individual query logs</li>
            </ul>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-lg font-medium">Data Management</h3>

          <div className="flex flex-col space-y-2">
            <Button variant="outline" className="justify-start">
              <Download className="h-4 w-4 mr-2" />
              Request Data Report
            </Button>

            <p className="text-sm text-muted-foreground mt-2">
              You can request a report of all available data related to your child's learning journey. This report will
              be delivered to your registered email address within 48 hours.
            </p>
          </div>
        </div>

        <div className="text-sm text-muted-foreground border-t pt-4">
          <p>
            For more information about our privacy practices, please visit{" "}
            <a href="#" className="text-primary underline">
              CurioNext Privacy Center
            </a>{" "}
            or contact our Data Protection Officer at privacy@curionext.com
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
