import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Shield, Info } from "lucide-react"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"

export function PrivacyPolicy() {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Shield className="h-5 w-5 text-primary" />
          <CardTitle>CurioNext Privacy Policy</CardTitle>
        </div>
        <CardDescription>Our standardized approach to protecting your child's privacy</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="bg-primary/10 p-4 rounded-lg flex items-start gap-3">
          <Info className="h-5 w-5 text-primary mt-0.5" />
          <div>
            <h3 className="font-medium text-sm">Privacy-First Approach</h3>
            <p className="text-sm text-muted-foreground mt-1">
              CurioNext follows a standardized privacy policy that cannot be modified by individual users. This ensures
              consistent protection of children's data across all users of the platform.
            </p>
          </div>
        </div>

        <Accordion type="single" collapsible className="w-full">
          <AccordionItem value="item-1">
            <AccordionTrigger className="text-sm font-medium">Data Collection Principles</AccordionTrigger>
            <AccordionContent className="text-sm text-muted-foreground">
              <ul className="list-disc pl-5 space-y-2">
                <li>Questions are analyzed but never stored in their raw form</li>
                <li>Only aggregated themes and patterns are saved</li>
                <li>Personal identifiers are never associated with question content</li>
                <li>All data is encrypted both in transit and at rest</li>
              </ul>
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="item-2">
            <AccordionTrigger className="text-sm font-medium">What Parents Can See</AccordionTrigger>
            <AccordionContent className="text-sm text-muted-foreground">
              <ul className="list-disc pl-5 space-y-2">
                <li>Interest categories and their relative distribution</li>
                <li>Learning patterns and trends over time</li>
                <li>AI-generated insights and recommendations</li>
                <li>Activity levels and engagement metrics</li>
                <li>Parents never see specific questions asked by their child</li>
              </ul>
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="item-3">
            <AccordionTrigger className="text-sm font-medium">Data Retention Policy</AccordionTrigger>
            <AccordionContent className="text-sm text-muted-foreground">
              <ul className="list-disc pl-5 space-y-2">
                <li>Raw question data is processed in real-time and immediately discarded</li>
                <li>Aggregated insights are retained for 6 months by default</li>
                <li>Parents can request complete data deletion at any time</li>
                <li>Inactive accounts have all data automatically purged after 12 months</li>
              </ul>
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="item-4">
            <AccordionTrigger className="text-sm font-medium">Third-Party Access</AccordionTrigger>
            <AccordionContent className="text-sm text-muted-foreground">
              <ul className="list-disc pl-5 space-y-2">
                <li>No data is ever sold to third parties</li>
                <li>Anonymized, aggregated data may be used to improve the service</li>
                <li>No advertising or marketing partners have access to any data</li>
                <li>Data is never shared with educational institutions without explicit consent</li>
              </ul>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        <div className="text-sm text-muted-foreground">
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
