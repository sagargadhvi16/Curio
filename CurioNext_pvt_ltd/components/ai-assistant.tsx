"use client"

import { useState, useRef, useEffect } from "react"
import { Bot, Send, X, Maximize2, Minimize2, HelpCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"

// Sample assistant responses
const sampleResponses = [
  {
    query: "how to add geo fence",
    response:
      "To add a GEO fence, go to Safety Settings and click on 'Add New Fence'. Enter the location name, address, and set the radius. You can also set a schedule for when the fence should be active.",
  },
  {
    query: "what is interest map",
    response:
      "The Interest Map shows how your child's curiosity is distributed across different areas. It visualizes which topics Avani is most interested in and how these interests have changed over time.",
  },
  {
    query: "how to check notifications",
    response:
      "You can check notifications by clicking on the bell icon in the top right corner or by visiting the Notifications page from the sidebar menu.",
  },
  {
    query: "what happens if device is removed",
    response:
      "If the device is removed from your child's wrist, you'll receive an immediate alert notification. This safety feature helps ensure the device stays with your child at all times. You can configure these alerts in the Safety Settings under 'Device Removal Alerts'.",
  },
  {
    query: "how to add emergency contact",
    response:
      "To add an emergency contact, go to Safety Settings, select the Emergency Contacts tab, and click 'Add Contact'. Fill in the contact details, set their relationship and priority level.",
  },
  {
    query: "product recommendations",
    response:
      "You can find personalized product recommendations for Avani in the 'Recommendations' section of the dashboard. These suggestions are tailored based on Avani's age (7 years) and interests like dinosaurs, space, and how things work. Products are priced in Indian Rupees (₹) and you can browse books (₹1,350-₹1,899), educational toys (₹2,195-₹3,299), and creative play items (₹1,499-₹3,699). You can purchase directly from the CurioNext Store or through our affiliate partners like Amazon and Flipkart.",
  },
  {
    query: "where to buy books",
    response:
      "You can purchase recommended books for Avani through our Recommendations page. We offer both direct sales through the CurioNext Store and affiliate links to trusted retailers like Amazon and Flipkart. All recommendations are priced in Indian Rupees (₹) and are curated based on Avani's interests and age.",
  },
  {
    query: "location tracking",
    response:
      "The Location Tracking feature allows you to monitor Avani's real-time location on an interactive map. You can define safe zones around specific locations like home, school, or grandparents' house. If Avani moves outside these designated safe zones, you'll receive immediate alerts. You can also view location history and manage privacy settings to control how location data is collected and stored.",
  },
  {
    query: "safe zones",
    response:
      "Safe Zones are predefined areas where your child is allowed to be. You can create these zones around locations like home, school, or other trusted places. Each safe zone has a customizable radius and can be scheduled for specific times (e.g., school hours). If your child leaves a safe zone, you'll receive an immediate alert notification.",
  },
  {
    query: "location privacy",
    response:
      "We take location privacy seriously. In the Location Tracking settings, you can configure who can view your child's location data, how long location history is stored, and whether to share location with emergency contacts. All location data is encrypted and stored securely, and you can delete the history at any time.",
  },
  {
    query: "preference analysis",
    response:
      "The Preference Analysis feature analyzes Avani's questions to infer likes and dislikes. It identifies keywords, categorizes topics, and evaluates sentiment to create a comprehensive profile of preferences. You can view top likes and dislikes, sentiment trends, and emerging interests like Avani's recent interest in painting. This helps you better understand Avani's evolving interests and tailor learning experiences accordingly. For privacy protection, we never display the actual questions asked by your child - only the analyzed topics and preferences.",
  },
  {
    query: "how to see what my child likes",
    response:
      "To see what Avani likes, visit the Preference Analysis page from the sidebar menu. This feature analyzes questions to identify topics with positive sentiment. You'll see a dashboard with likes, dislikes, sentiment distribution, and topic trends. For privacy protection, we never display the actual questions asked by your child - only the analyzed topics and preferences. The system updates dynamically as Avani asks more questions, ensuring you always have current insights into their interests.",
  },
  {
    query: "privacy of questions",
    response:
      "We take your child's privacy very seriously. The Preference Analysis feature never displays the actual questions asked by your child. Instead, it shows only the analyzed topics, sentiment, and trends. All analysis is performed on-device and no raw question data is stored long-term. Only aggregated insights are saved to protect Avani's privacy while providing valuable information about their interests and preferences.",
  },
  {
    query: "painting interest",
    response:
      "Avani has recently developed a strong interest in painting! This new interest emerged in the last week and shows a high positive sentiment score of 0.90. The Preference Analysis dashboard has automatically detected this trend and updated to show painting as one of Avani's top interests. You can see related topics like colors, canvas, brushes, and techniques that Avani is curious about. The system dynamically updates as new interests emerge, ensuring you always have the most current insights into your child's preferences.",
  },
]

interface Message {
  id: number
  content: string
  sender: "user" | "assistant"
  timestamp: Date
}

export function AiAssistant() {
  const [isOpen, setIsOpen] = useState(false)
  const [isMinimized, setIsMinimized] = useState(false)
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      content: "Hi there! I'm your CurioNext assistant. How can I help you with the dashboard today?",
      sender: "assistant",
      timestamp: new Date(),
    },
  ])
  const [inputValue, setInputValue] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Suggested questions
  const suggestedQuestions = [
    "How do I track my child's location?",
    "What are safe zones?",
    "How can I see what my child likes?",
    "Where can I find product recommendations?",
    "How does preference analysis work?",
  ]

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [messages])

  const handleSend = () => {
    if (inputValue.trim() === "") return

    // Add user message
    const userMessage: Message = {
      id: messages.length + 1,
      content: inputValue,
      sender: "user",
      timestamp: new Date(),
    }
    setMessages([...messages, userMessage])
    setInputValue("")
    setIsTyping(true)

    // Simulate assistant response
    setTimeout(() => {
      // Find a matching response or use a default
      const lowerQuery = inputValue.toLowerCase()
      const matchedResponse = sampleResponses.find((item) => lowerQuery.includes(item.query.toLowerCase()))

      const assistantMessage: Message = {
        id: messages.length + 2,
        content: matchedResponse
          ? matchedResponse.response
          : "I don't have specific information about that yet. Please check the dashboard sections or try asking something else.",
        sender: "assistant",
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, assistantMessage])
      setIsTyping(false)
    }, 1500)
  }

  const handleSuggestedQuestion = (question: string) => {
    setInputValue(question)
    handleSend()
  }

  if (!isOpen) {
    return (
      <Button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 right-4 h-12 w-12 rounded-full shadow-lg bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-600 hover:to-amber-700"
      >
        <HelpCircle className="h-6 w-6" />
      </Button>
    )
  }

  if (isMinimized) {
    return (
      <div className="fixed bottom-4 right-4 z-50">
        <Button
          onClick={() => setIsMinimized(false)}
          className="flex h-12 items-center gap-2 rounded-full shadow-lg bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-600 hover:to-amber-700"
        >
          <Bot className="h-5 w-5" />
          <span>CurioNext Assistant</span>
          <Maximize2 className="h-4 w-4 ml-1" />
        </Button>
      </div>
    )
  }

  return (
    <Card className="fixed bottom-4 right-4 w-80 md:w-96 shadow-lg z-50 border-amber-200 dark:border-amber-800">
      <CardHeader className="flex flex-row items-center justify-between p-3 border-b bg-gradient-to-r from-amber-50 to-amber-100 dark:from-amber-950 dark:to-amber-900">
        <div className="flex items-center gap-2">
          <Avatar className="h-8 w-8 border border-amber-200 dark:border-amber-800">
            <AvatarImage
              src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/curionext-high-resolution-logo-UlyGnnH7xKKOIsZUrsuoO2gVadu0rT.png"
              alt="CurioNext Assistant"
            />
            <AvatarFallback>CN</AvatarFallback>
          </Avatar>
          <div>
            <h3 className="font-medium text-sm">CurioNext Assistant</h3>
            <p className="text-xs text-muted-foreground">Here to help</p>
          </div>
        </div>
        <div className="flex gap-1">
          <Button variant="ghost" size="icon" onClick={() => setIsMinimized(true)} className="h-7 w-7">
            <Minimize2 className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" onClick={() => setIsOpen(false)} className="h-7 w-7">
            <X className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="h-80 overflow-y-auto p-3 space-y-4">
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[80%] rounded-lg p-3 ${
                  message.sender === "user" ? "bg-gradient-to-r from-amber-500 to-amber-600 text-white" : "bg-muted"
                }`}
              >
                <p className="text-sm">{message.content}</p>
                <p className="text-xs opacity-70 mt-1">
                  {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </p>
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="flex justify-start">
              <div className="max-w-[80%] rounded-lg p-3 bg-muted">
                <div className="flex space-x-1">
                  <div className="h-2 w-2 rounded-full bg-amber-400 animate-bounce [animation-delay:0ms]"></div>
                  <div className="h-2 w-2 rounded-full bg-amber-500 animate-bounce [animation-delay:150ms]"></div>
                  <div className="h-2 w-2 rounded-full bg-amber-600 animate-bounce [animation-delay:300ms]"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {messages.length < 3 && (
          <div className="px-3 pb-2">
            <p className="text-xs text-muted-foreground mb-2">Suggested questions:</p>
            <div className="flex flex-wrap gap-1">
              {suggestedQuestions.map((question, index) => (
                <Badge
                  key={index}
                  variant="outline"
                  className="cursor-pointer hover:bg-muted border-amber-200 dark:border-amber-800"
                  onClick={() => handleSuggestedQuestion(question)}
                >
                  {question}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
      <CardFooter className="p-2 border-t">
        <div className="flex w-full items-center gap-2">
          <Input
            placeholder="Type your question..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                handleSend()
              }
            }}
            className="h-9 border-amber-200 dark:border-amber-800 focus-visible:ring-amber-500"
          />
          <Button
            size="icon"
            onClick={handleSend}
            className="h-9 w-9 bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-600 hover:to-amber-700"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </CardFooter>
    </Card>
  )
}
