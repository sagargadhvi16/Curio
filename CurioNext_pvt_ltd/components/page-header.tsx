import { Button } from "@/components/ui/button"
import { ArrowLeft } from "lucide-react"
import Link from "next/link"

interface PageHeaderProps {
  title: string
  description?: string
  backLink?: string
}

export function PageHeader({ title, description, backLink = "/" }: PageHeaderProps) {
  return (
    <div className="flex flex-col gap-1 mb-6">
      <div className="flex items-center gap-2">
        {backLink && (
          <Button variant="ghost" size="icon" asChild className="mr-2">
            <Link href={backLink}>
              <ArrowLeft className="h-4 w-4" />
              <span className="sr-only">Back</span>
            </Link>
          </Button>
        )}
        <h1 className="text-2xl font-bold">{title}</h1>
      </div>
      {description && <p className="text-muted-foreground">{description}</p>}
    </div>
  )
}
