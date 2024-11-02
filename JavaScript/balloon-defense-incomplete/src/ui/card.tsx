import * as React from "react"
import { cn } from "../lib/utils"  // Updated import path

const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "rounded-xl border bg-white text-gray-900 shadow p-4",
      className
    )}
    {...props}
  />
))
Card.displayName = "Card"

export { Card }
