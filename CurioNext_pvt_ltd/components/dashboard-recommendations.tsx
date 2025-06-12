"use client"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Star, ShoppingBag, ArrowRight, BookOpen, Puzzle, Palette } from "lucide-react"
import Image from "next/image"
import Link from "next/link"

// Sample featured products for Avani
const featuredProducts = [
  {
    id: 1,
    name: "The Magic of Dinosaurs: An Interactive Journey",
    price: 1899,
    rating: 4.8,
    image: "/placeholder.svg?height=200&width=200",
    category: "Dinosaurs",
    link: "#",
    description: "Explore the prehistoric world with this interactive book featuring pop-ups and sound effects.",
    ageRange: "6-9 years",
    store: "Amazon",
    isAffiliate: true,
  },
  {
    id: 2,
    name: "Build-A-Dino Construction Set",
    price: 2499,
    rating: 4.7,
    image: "/placeholder.svg?height=200&width=200",
    category: "Dinosaurs",
    link: "#",
    description: "Create your own dinosaur models with this 200-piece educational construction set.",
    ageRange: "6-10 years",
    store: "CurioNext Store",
    isAffiliate: false,
  },
  {
    id: 3,
    name: "Space Explorer's Handbook",
    price: 1499,
    rating: 4.7,
    image: "/placeholder.svg?height=200&width=200",
    category: "Space",
    link: "#",
    description: "A comprehensive guide to space exploration with stunning illustrations and fascinating facts.",
    ageRange: "7-10 years",
    store: "CurioNext Store",
    isAffiliate: false,
  },
]

// Helper function to format price in INR
const formatPrice = (price: number) => {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(price)
}

// Helper function to get icon based on category
const getCategoryIcon = (category: string) => {
  switch (category.toLowerCase()) {
    case "books":
    case "reading":
      return <BookOpen className="h-4 w-4" />
    case "toys":
    case "puzzles":
      return <Puzzle className="h-4 w-4" />
    case "creative":
    case "art":
      return <Palette className="h-4 w-4" />
    default:
      return null
  }
}

export function DashboardRecommendations() {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Recommended for Avani</CardTitle>
          <CardDescription>Based on recent interests in dinosaurs and space</CardDescription>
        </div>
        <Button variant="outline" size="sm" asChild>
          <Link href="/recommendations">
            View All
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {featuredProducts.map((product) => (
            <div
              key={product.id}
              className="relative group overflow-hidden rounded-lg border bg-card transition-all hover:shadow-md"
            >
              <div className="aspect-[4/3] relative overflow-hidden">
                <Image
                  src={product.image || "/placeholder.svg"}
                  alt={product.name}
                  fill
                  className="object-cover transition-transform group-hover:scale-105"
                />
                <Badge className="absolute top-2 right-2 bg-primary/90">{product.category}</Badge>
                <Badge className="absolute top-2 left-2 bg-amber-500/90">{product.ageRange}</Badge>
              </div>
              <div className="p-4">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-bold text-amber-600">{formatPrice(product.price)}</span>
                  <div className="flex items-center text-amber-500">
                    <Star className="h-3 w-3 fill-current" />
                    <span className="text-xs ml-1">{product.rating}</span>
                  </div>
                </div>
                <h3 className="font-medium text-base line-clamp-1 mb-1">{product.name}</h3>
                <p className="text-sm text-muted-foreground line-clamp-2 mb-2">{product.description}</p>
                <div className="flex items-center justify-between">
                  <Badge variant={product.isAffiliate ? "outline" : "secondary"} className="text-xs">
                    {product.store}
                  </Badge>
                  <Button variant="ghost" size="sm" className="h-7 px-2">
                    View Details
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
      <CardFooter className="border-t pt-4">
        <Button asChild className="w-full">
          <Link href="/recommendations">
            <ShoppingBag className="mr-2 h-4 w-4" />
            Browse All Recommendations
          </Link>
        </Button>
      </CardFooter>
    </Card>
  )
}
