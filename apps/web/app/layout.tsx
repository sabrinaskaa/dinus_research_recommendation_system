import "./globals.css";

export const metadata = {
  title: "Research Recommendation Demo",
  description: "IR + Data Mining demo",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="id">
      <body>{children}</body>
    </html>
  );
}
