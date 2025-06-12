"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Star, ShoppingCart, ExternalLink, BookOpen, Puzzle, Palette, Filter, X } from "lucide-react"
import Image from "next/image"
import Link from "next/link"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

// Sample product data for Avani (7 years old)
const products = {
  books: [
    {
      id: 1,
      name: "The Magic of Dinosaurs: An Interactive Journey",
      description:
        "Explore the prehistoric world with this interactive book featuring pop-ups and sound effects. This beautifully illustrated book brings dinosaurs to life with interactive elements that make learning fun.",
      price: 1899,
      rating: 4.8,
      image: "/placeholder.svg?height=200&width=200",
      category: "Dinosaurs",
      ageRange: "6-9 years",
      link: "#",
      isAffiliate: true,
      store: "Amazon",
      skills: ["Reading", "Science", "History"],
      milestones: ["Cognitive Development", "Knowledge Retention"],
    },
    {
      id: 2,
      name: "Space Explorer's Handbook",
      description:
        "A comprehensive guide to space exploration with stunning illustrations and fascinating facts. Perfect for young astronomers who want to learn about planets, stars, and space missions.",
      price: 1499,
      rating: 4.7,
      image: "/placeholder.svg?height=200&width=200",
      category: "Space",
      ageRange: "7-10 years",
      link: "#",
      isAffiliate: false,
      store: "CurioNext Store",
      skills: ["Science", "Astronomy", "Reading"],
      milestones: ["Scientific Thinking", "Curiosity Development"],
    },
    {
      id: 3,
      name: "Earth's Mighty Volcanoes",
      description:
        "Discover the science behind volcanoes with this beautifully illustrated educational book. Learn about different types of volcanoes, famous eruptions, and how they shape our planet.",
      price: 1350,
      rating: 4.6,
      image: "/placeholder.svg?height=200&width=200",
      category: "Volcanoes",
      ageRange: "6-8 years",
      link: "#",
      isAffiliate: true,
      store: "Flipkart",
      skills: ["Science", "Geography", "Reading"],
      milestones: ["Earth Science Understanding", "Cause and Effect"],
    },
    {
      id: 4,
      name: "The Big Book of Why: Science Edition",
      description:
        "Answers to hundreds of questions about how things work, perfect for curious minds. This book tackles the most common 'why' questions children ask about science and the world around them.",
      price: 1699,
      rating: 4.9,
      image: "/placeholder.svg?height=200&width=200",
      category: "How Things Work",
      ageRange: "7-12 years",
      link: "#",
      isAffiliate: true,
      store: "Amazon",
      skills: ["Critical Thinking", "Science", "Reading"],
      milestones: ["Question Formulation", "Analytical Thinking"],
    },
  ],
  toys: [
    {
      id: 1,
      name: "Build-A-Dino Construction Set",
      description:
        "Create your own dinosaur models with this 200-piece educational construction set. Includes detailed instructions for building 5 different dinosaur species with movable parts.",
      price: 2499,
      rating: 4.7,
      image: "/placeholder.svg?height=200&width=200",
      category: "Dinosaurs",
      ageRange: "6-10 years",
      link: "#",
      isAffiliate: false,
      store: "CurioNext Store",
      skills: ["Fine Motor Skills", "Following Instructions", "Creativity"],
      milestones: ["Spatial Reasoning", "Patience Development"],
    },
    {
      id: 2,
      name: "Solar System 3D Puzzle",
      description:
        "A challenging 3D puzzle that creates a detailed model of our solar system. When completed, the planets can be illuminated with the included LED base.",
      price: 2195,
      rating: 4.5,
      image: "/placeholder.svg?height=200&width=200",
      category: "Space",
      ageRange: "7-12 years",
      link: "#",
      isAffiliate: true,
      store: "Firstcry",
      skills: ["Problem Solving", "Astronomy", "Fine Motor Skills"],
      milestones: ["Spatial Awareness", "Persistence"],
    },
    {
      id: 3,
      name: "Volcano Science Lab Kit",
      description:
        "Create erupting volcanoes while learning about chemistry and earth science. This kit includes all materials needed for multiple eruption experiments and a detailed learning guide.",
      price: 2899,
      rating: 4.8,
      image: "/placeholder.svg?height=200&width=200",
      category: "Volcanoes",
      ageRange: "7-14 years",
      link: "#",
      isAffiliate: true,
      store: "Amazon",
      skills: ["Science", "Following Instructions", "Observation"],
      milestones: ["Scientific Method", "Cause and Effect"],
    },
    {
      id: 4,
      name: "Junior Engineer Mechanics Kit",
      description:
        "Build simple machines and learn about physics with this hands-on engineering kit. Includes gears, pulleys, levers, and wheels to create over 10 different working models.",
      price: 3299,
      rating: 4.9,
      image: "/placeholder.svg?height=200&width=200",
      category: "How Things Work",
      ageRange: "6-9 years",
      link: "#",
      isAffiliate: false,
      store: "CurioNext Store",
      skills: ["Engineering", "Problem Solving", "Fine Motor Skills"],
      milestones: ["Mechanical Understanding", "Logical Thinking"],
    },
  ],
  creative: [
    {
      id: 1,
      name: "Prehistoric Art Studio",
      description:
        "Create dinosaur-themed art with this complete art studio set including paints, clay, and more. Includes stencils, stamps, and a guidebook with dinosaur facts and art techniques.",
      price: 2099,
      rating: 4.6,
      image: "/placeholder.svg?height=200&width=200",
      category: "Dinosaurs",
      ageRange: "5-10 years",
      link: "#",
      isAffiliate: true,
      store: "Flipkart",
      skills: ["Creativity", "Fine Motor Skills", "Color Recognition"],
      milestones: ["Artistic Expression", "Attention to Detail"],
    },
    {
      id: 2,
      name: "Glow-in-the-Dark Stars & Planets Kit",
      description:
        "Transform any room into a cosmic wonderland with these realistic glow-in-the-dark celestial bodies. Includes over 200 stars, all planets, and a moon with adhesive backing.",
      price: 1499,
      rating: 4.7,
      image: "/placeholder.svg?height=200&width=200",
      category: "Space",
      ageRange: "5+ years",
      link: "#",
      isAffiliate: true,
      store: "Amazon",
      skills: ["Astronomy", "Spatial Awareness", "Creativity"],
      milestones: ["Pattern Recognition", "Cosmic Understanding"],
    },
    {
      id: 3,
      name: "Nature Explorer's Journal Set",
      description:
        "Document discoveries with this beautiful journal set including colored pencils and field guides. Perfect for outdoor adventures and nature observations.",
      price: 1795,
      rating: 4.8,
      image: "/placeholder.svg?height=200&width=200",
      category: "Nature & Life",
      ageRange: "6-12 years",
      link: "#",
      isAffiliate: false,
      store: "CurioNext Store",
      skills: ["Observation", "Writing", "Drawing"],
      milestones: ["Scientific Documentation", "Nature Appreciation"],
    },
    {
      id: 4,
      name: "Young Inventor's Workshop",
      description:
        "Unleash creativity with this kit containing materials to build simple inventions and machines. Includes motors, LEDs, switches, and recycled materials for unlimited creations.",
      price: 3699,
      rating: 4.9,
      image: "/placeholder.svg?height=200&width=200",
      category: "How Things Work",
      ageRange: "7-12 years",
      link: "#",
      isAffiliate: true,
      store: "Firstcry",
      skills: ["Engineering", "Creativity", "Problem Solving"],
      milestones: ["Innovation", "Design Thinking"],
    },
  ],
}

// Helper function to format price in INR
const formatPrice = (price: number) => {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(price)
}

type ProductCategory = "books" | "toys" | "creative"

interface ProductCardProps {
  product: {
    id: number
    name: string
    description: string
    price: number
    rating: number
    image: string
    category: string
    ageRange: string
    link: string
    isAffiliate: boolean
    store: string
    skills?: string[]
    milestones?: string[]
  }
}

function ProductCard({ product }: ProductCardProps) {
  return (
    <Card className="h-full flex flex-col overflow-hidden transition-all hover:shadow-md">
      <div className="relative aspect-square bg-muted/50 flex items-center justify-center overflow-hidden">
        <Image
          src={product.image || "/placeholder.svg"}
          alt={product.name}
          width={200}
          height={200}
          className="object-cover w-full h-full"
        />
        <Badge className="absolute top-2 right-2 bg-primary/90">{product.category}</Badge>
        <Badge className="absolute top-2 left-2 bg-amber-500/90">{product.ageRange}</Badge>
      </div>
      <CardContent className="flex-1 flex flex-col p-4">
        <div className="mb-2 flex items-center justify-between">
          <span className="font-bold text-lg text-amber-600">{formatPrice(product.price)}</span>
          <div className="flex items-center text-amber-500">
            <Star className="h-4 w-4 fill-current" />
            <span className="text-sm ml-1">{product.rating}</span>
          </div>
        </div>
        <h3 className="font-semibold text-lg line-clamp-2 mb-2">{product.name}</h3>
        <p className="text-sm text-muted-foreground line-clamp-3 mb-3 flex-1">{product.description}</p>

        {product.skills && product.skills.length > 0 && (
          <div className="mb-2">
            <p className="text-xs text-muted-foreground mb-1">Skills Developed:</p>
            <div className="flex flex-wrap gap-1">
              {product.skills.map((skill, index) => (
                <Badge key={index} variant="outline" className="text-xs">
                  {skill}
                </Badge>
              ))}
            </div>
          </div>
        )}

        <div className="mt-auto">
          <div className="flex items-center justify-between mb-3">
            <Badge variant={product.isAffiliate ? "secondary" : "default"} className="text-xs">
              {product.store}
            </Badge>
            {product.milestones && product.milestones.length > 0 && (
              <div className="text-xs text-muted-foreground">Supports: {product.milestones.join(", ")}</div>
            )}
          </div>
          <Button asChild className="w-full gap-2">
            <Link href={product.link}>
              {product.isAffiliate ? <ExternalLink className="h-4 w-4" /> : <ShoppingCart className="h-4 w-4" />}
              {product.isAffiliate ? "View on " + product.store : "Add to Cart"}
            </Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

export function ProductRecommendations() {
  const [category, setCategory] = useState<ProductCategory>("books")
  const [filter, setFilter] = useState<string | null>(null)
  const [ageFilter, setAgeFilter] = useState<string | null>(null)
  const [priceRange, setPriceRange] = useState<string | null>(null)
  const [showFilters, setShowFilters] = useState(false)

  // Get unique categories from products
  const getUniqueCategories = () => {
    const allProducts = [...products.books, ...products.toys, ...products.creative]
    return [...new Set(allProducts.map((product) => product.category))]
  }

  // Get unique age ranges
  const getUniqueAgeRanges = () => {
    const allProducts = [...products.books, ...products.toys, ...products.creative]
    return [...new Set(allProducts.map((product) => product.ageRange))]
  }

  const uniqueCategories = getUniqueCategories()
  const uniqueAgeRanges = getUniqueAgeRanges()

  // Filter products by interest category, age range, and price
  const filteredProducts = products[category].filter((product) => {
    // Filter by interest category
    if (filter && product.category !== filter) {
      return false
    }

    // Filter by age range
    if (ageFilter && product.ageRange !== ageFilter) {
      return false
    }

    // Filter by price range
    if (priceRange) {
      const price = product.price
      if (priceRange === "under1000" && price >= 1000) return false
      if (priceRange === "1000to2000" && (price < 1000 || price > 2000)) return false
      if (priceRange === "2000to3000" && (price < 2000 || price > 3000)) return false
      if (priceRange === "above3000" && price <= 3000) return false
    }

    return true
  })

  const clearFilters = () => {
    setFilter(null)
    setAgeFilter(null)
    setPriceRange(null)
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <CardTitle>Recommended for Avani</CardTitle>
            <CardDescription>Curated products based on Avani's interests and age (7 years)</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center gap-2"
            >
              {showFilters ? <X className="h-4 w-4" /> : <Filter className="h-4 w-4" />}
              {showFilters ? "Hide Filters" : "Show Filters"}
            </Button>
          </div>
        </div>

        {showFilters && (
          <div className="mt-4 p-4 bg-muted/30 rounded-lg space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="text-sm font-medium mb-1 block">Interest Category</label>
                <Select value={filter || ""} onValueChange={(value) => setFilter(value || null)}>
                  <SelectTrigger>
                    <SelectValue placeholder="All Categories" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">All Categories</SelectItem>
                    {uniqueCategories.map((cat) => (
                      <SelectItem key={cat} value={cat}>
                        {cat}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-1 block">Age Range</label>
                <Select value={ageFilter || ""} onValueChange={(value) => setAgeFilter(value || null)}>
                  <SelectTrigger>
                    <SelectValue placeholder="All Ages" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">All Ages</SelectItem>
                    {uniqueAgeRanges.map((age) => (
                      <SelectItem key={age} value={age}>
                        {age}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-1 block">Price Range</label>
                <Select value={priceRange || ""} onValueChange={(value) => setPriceRange(value || null)}>
                  <SelectTrigger>
                    <SelectValue placeholder="All Prices" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">All Prices</SelectItem>
                    <SelectItem value="under1000">Under ₹1,000</SelectItem>
                    <SelectItem value="1000to2000">₹1,000 - ₹2,000</SelectItem>
                    <SelectItem value="2000to3000">₹2,000 - ₹3,000</SelectItem>
                    <SelectItem value="above3000">Above ₹3,000</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="flex justify-end">
              <Button variant="outline" size="sm" onClick={clearFilters}>
                Clear All Filters
              </Button>
            </div>
          </div>
        )}
      </CardHeader>
      <CardContent>
        <Tabs value={category} onValueChange={(value) => setCategory(value as ProductCategory)}>
          <TabsList className="mb-6">
            <TabsTrigger value="books" className="flex items-center gap-2">
              <BookOpen className="h-4 w-4" />
              Books
            </TabsTrigger>
            <TabsTrigger value="toys" className="flex items-center gap-2">
              <Puzzle className="h-4 w-4" />
              Educational Toys
            </TabsTrigger>
            <TabsTrigger value="creative" className="flex items-center gap-2">
              <Palette className="h-4 w-4" />
              Creative Play
            </TabsTrigger>
          </TabsList>

          {["books", "toys", "creative"].map((cat) => (
            <TabsContent key={cat} value={cat}>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {filteredProducts.map((product) => (
                  <ProductCard key={product.id} product={product} />
                ))}
              </div>
              {filteredProducts.length === 0 && (
                <div className="text-center py-12">
                  <p className="text-muted-foreground">No products found matching your filters.</p>
                  <Button variant="outline" className="mt-4" onClick={clearFilters}>
                    Clear Filters
                  </Button>
                </div>
              )}
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  )
}
