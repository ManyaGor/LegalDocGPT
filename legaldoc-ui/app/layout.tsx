// app/layout.tsx
import type { ReactNode } from "react"
import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import { Suspense } from "react"
import "./globals.css"

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "LegalDocGPT - AI-Powered Legal Document Analysis",
  description:
    "Boost your legal analysis with AI. Transform complex legal documents into simple, understandable summaries using advanced AI technology.",
  generator: "v0.app",
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
       <body className={`font-sans ${geistSans.variable} ${geistMono.variable}`}>
         <Suspense fallback={<div>Loading...</div>}>{children}</Suspense>
       </body>
    </html>
  )
}
