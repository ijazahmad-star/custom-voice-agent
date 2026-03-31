import type { Metadata } from "next";
import { Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-outfit",
});

export const metadata: Metadata = {
  title: "Aura - AI Voice Agent",
  description: "Next-generation interactive voice assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${outfit.variable} font-sans h-full antialiased dark`}
      suppressHydrationWarning
    >
      <body className="min-h-full bg-[#0a0a0a] text-white flex flex-col" suppressHydrationWarning>{children}</body>
    </html>
  );
}
