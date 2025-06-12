"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Phone, Plus, Trash2, Save, User, Heart, Shield, Edit, AlertCircle } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

// Sample emergency contacts data
const initialContacts = [
  {
    id: 1,
    name: "Mom",
    phone: "+1 (555) 123-4567",
    relationship: "Parent",
    priority: "primary",
    avatar: "/placeholder.svg?height=40&width=40",
    initials: "MM",
  },
  {
    id: 2,
    name: "Dad",
    phone: "+1 (555) 987-6543",
    relationship: "Parent",
    priority: "primary",
    avatar: "/placeholder.svg?height=40&width=40",
    initials: "DD",
  },
  {
    id: 3,
    name: "Grandma",
    phone: "+1 (555) 456-7890",
    relationship: "Grandparent",
    priority: "secondary",
    avatar: "/placeholder.svg?height=40&width=40",
    initials: "GM",
  },
  {
    id: 4,
    name: "Emergency Services",
    phone: "911",
    relationship: "Emergency",
    priority: "emergency",
    avatar: "/placeholder.svg?height=40&width=40",
    initials: "ES",
  },
]

export function EmergencyContacts() {
  const [contacts, setContacts] = useState(initialContacts)
  const [newContact, setNewContact] = useState({
    name: "",
    phone: "",
    relationship: "Parent",
    priority: "secondary",
    avatar: "/placeholder.svg?height=40&width=40",
    initials: "",
  })
  const [editingContact, setEditingContact] = useState<number | null>(null)
  const [activeTab, setActiveTab] = useState("list")
  const [childView, setChildView] = useState(false)

  const handleDelete = (id: number) => {
    setContacts(contacts.filter((contact) => contact.id !== id))
  }

  const handleEdit = (id: number) => {
    const contact = contacts.find((c) => c.id === id)
    if (contact) {
      setNewContact({
        name: contact.name,
        phone: contact.phone,
        relationship: contact.relationship,
        priority: contact.priority,
        avatar: contact.avatar,
        initials: contact.initials,
      })
      setEditingContact(id)
      setActiveTab("add")
    }
  }

  const handleSave = () => {
    // Generate initials if not provided
    const initials =
      newContact.initials ||
      newContact.name
        .split(" ")
        .map((n) => n[0])
        .join("")
        .toUpperCase()

    if (editingContact) {
      setContacts(
        contacts.map((contact) =>
          contact.id === editingContact
            ? {
                ...contact,
                name: newContact.name,
                phone: newContact.phone,
                relationship: newContact.relationship,
                priority: newContact.priority,
                initials: initials,
              }
            : contact,
        ),
      )
      setEditingContact(null)
    } else {
      const newId = Math.max(0, ...contacts.map((c) => c.id)) + 1
      setContacts([
        ...contacts,
        {
          id: newId,
          name: newContact.name,
          phone: newContact.phone,
          relationship: newContact.relationship,
          priority: newContact.priority,
          avatar: "/placeholder.svg?height=40&width=40",
          initials: initials,
        },
      ])
    }

    // Reset form
    setNewContact({
      name: "",
      phone: "",
      relationship: "Parent",
      priority: "secondary",
      avatar: "/placeholder.svg?height=40&width=40",
      initials: "",
    })
    setActiveTab("list")
  }

  const handleCancel = () => {
    setNewContact({
      name: "",
      phone: "",
      relationship: "Parent",
      priority: "secondary",
      avatar: "/placeholder.svg?height=40&width=40",
      initials: "",
    })
    setEditingContact(null)
    setActiveTab("list")
  }

  const getPriorityBadge = (priority: string) => {
    switch (priority) {
      case "primary":
        return <Badge className="bg-green-500">Primary</Badge>
      case "secondary":
        return <Badge variant="secondary">Secondary</Badge>
      case "emergency":
        return <Badge className="bg-red-500">Emergency</Badge>
      default:
        return <Badge variant="outline">Other</Badge>
    }
  }

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case "primary":
        return <Heart className="h-4 w-4 text-green-500" />
      case "secondary":
        return <User className="h-4 w-4 text-blue-500" />
      case "emergency":
        return <AlertCircle className="h-4 w-4 text-red-500" />
      default:
        return <User className="h-4 w-4" />
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Emergency Contacts</CardTitle>
            <CardDescription>Manage emergency contacts that your child can access</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center space-x-2">
              <Switch id="child-view" checked={childView} onCheckedChange={setChildView} />
              <Label htmlFor="child-view">Child View</Label>
            </div>
            {!childView && (
              <Button variant="outline" onClick={() => setActiveTab("add")}>
                <Plus className="h-4 w-4 mr-2" />
                Add Contact
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {childView ? (
          <div className="space-y-4">
            <div className="bg-muted/30 p-4 rounded-lg flex items-start gap-3">
              <Shield className="h-5 w-5 text-primary mt-0.5" />
              <div>
                <h3 className="font-medium text-sm">Child's Emergency Contact View</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  This is what your child will see when they access emergency contacts on their device.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {contacts
                .sort((a, b) => {
                  const priorityOrder = { primary: 0, secondary: 1, emergency: 2 }
                  return (
                    priorityOrder[a.priority as keyof typeof priorityOrder] -
                    priorityOrder[b.priority as keyof typeof priorityOrder]
                  )
                })
                .map((contact) => (
                  <Button
                    key={contact.id}
                    variant="outline"
                    className="h-auto py-6 flex flex-col items-center justify-center gap-2"
                  >
                    <Avatar className="h-16 w-16">
                      <AvatarImage src={contact.avatar} alt={contact.name} />
                      <AvatarFallback>{contact.initials}</AvatarFallback>
                    </Avatar>
                    <div className="text-center">
                      <div className="font-medium text-lg">{contact.name}</div>
                      <div className="text-sm text-muted-foreground">{contact.phone}</div>
                    </div>
                    <div className="flex items-center gap-1 mt-1">
                      {getPriorityIcon(contact.priority)}
                      <span className="text-xs">{contact.relationship}</span>
                    </div>
                  </Button>
                ))}
            </div>

            <div className="mt-4 text-center">
              <p className="text-sm text-muted-foreground">
                Your child can tap any contact to call them in case of emergency.
              </p>
            </div>
          </div>
        ) : (
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="mb-4">
              <TabsTrigger value="list">Contact List</TabsTrigger>
              <TabsTrigger value="add">{editingContact ? "Edit" : "Add"} Contact</TabsTrigger>
            </TabsList>

            <TabsContent value="list">
              <div className="space-y-4">
                {contacts.length === 0 ? (
                  <div className="text-center py-8">
                    <Phone className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground">No emergency contacts added yet</p>
                    <Button variant="outline" className="mt-4" onClick={() => setActiveTab("add")}>
                      <Plus className="h-4 w-4 mr-2" />
                      Add Your First Contact
                    </Button>
                  </div>
                ) : (
                  contacts.map((contact) => (
                    <div key={contact.id} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <Avatar>
                            <AvatarImage src={contact.avatar} alt={contact.name} />
                            <AvatarFallback>{contact.initials}</AvatarFallback>
                          </Avatar>
                          <div>
                            <div className="flex items-center gap-2">
                              <h3 className="font-medium">{contact.name}</h3>
                              {getPriorityBadge(contact.priority)}
                            </div>
                            <p className="text-sm text-muted-foreground mt-1">{contact.phone}</p>
                            <p className="text-xs text-muted-foreground mt-1">{contact.relationship}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button variant="ghost" size="icon" onClick={() => handleEdit(contact.id)}>
                            <Edit className="h-4 w-4" />
                          </Button>
                          <Dialog>
                            <DialogTrigger asChild>
                              <Button variant="ghost" size="icon">
                                <Trash2 className="h-4 w-4 text-destructive" />
                              </Button>
                            </DialogTrigger>
                            <DialogContent>
                              <DialogHeader>
                                <DialogTitle>Delete Contact</DialogTitle>
                                <DialogDescription>
                                  Are you sure you want to delete {contact.name} from emergency contacts? This action
                                  cannot be undone.
                                </DialogDescription>
                              </DialogHeader>
                              <DialogFooter>
                                <Button variant="outline" onClick={() => {}}>
                                  Cancel
                                </Button>
                                <Button variant="destructive" onClick={() => handleDelete(contact.id)}>
                                  Delete
                                </Button>
                              </DialogFooter>
                            </DialogContent>
                          </Dialog>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </TabsContent>

            <TabsContent value="add">
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">Contact Name</Label>
                    <Input
                      id="name"
                      placeholder="e.g., Mom, Dad, Grandma"
                      value={newContact.name}
                      onChange={(e) => setNewContact({ ...newContact, name: e.target.value })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="phone">Phone Number</Label>
                    <Input
                      id="phone"
                      placeholder="e.g., +1 (555) 123-4567"
                      value={newContact.phone}
                      onChange={(e) => setNewContact({ ...newContact, phone: e.target.value })}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="relationship">Relationship</Label>
                    <Select
                      value={newContact.relationship}
                      onValueChange={(value) => setNewContact({ ...newContact, relationship: value })}
                    >
                      <SelectTrigger id="relationship">
                        <SelectValue placeholder="Select relationship" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Parent">Parent</SelectItem>
                        <SelectItem value="Grandparent">Grandparent</SelectItem>
                        <SelectItem value="Sibling">Sibling</SelectItem>
                        <SelectItem value="Relative">Other Relative</SelectItem>
                        <SelectItem value="Friend">Family Friend</SelectItem>
                        <SelectItem value="Emergency">Emergency Services</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="priority">Priority Level</Label>
                    <Select
                      value={newContact.priority}
                      onValueChange={(value) => setNewContact({ ...newContact, priority: value })}
                    >
                      <SelectTrigger id="priority">
                        <SelectValue placeholder="Select priority" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="primary">Primary (First to call)</SelectItem>
                        <SelectItem value="secondary">Secondary</SelectItem>
                        <SelectItem value="emergency">Emergency Services</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="initials">Initials (Optional)</Label>
                  <Input
                    id="initials"
                    placeholder="e.g., MM for Mom"
                    value={newContact.initials}
                    onChange={(e) => setNewContact({ ...newContact, initials: e.target.value })}
                    maxLength={2}
                  />
                  <p className="text-xs text-muted-foreground">Leave blank to auto-generate from the name</p>
                </div>

                <div className="flex justify-end gap-2 mt-6">
                  <Button variant="outline" onClick={handleCancel}>
                    Cancel
                  </Button>
                  <Button onClick={handleSave} disabled={!newContact.name || !newContact.phone}>
                    <Save className="h-4 w-4 mr-2" />
                    {editingContact ? "Update" : "Save"} Contact
                  </Button>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        )}
      </CardContent>
    </Card>
  )
}
