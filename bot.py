import os
import ssl
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from prediction_engine import PredictionEngine
from chart_generator import ChartGenerator
from news_analyzer import NewsAnalyzer
import asyncio

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$sp ', intents=intents, help_command=None)

ADMIN_USER_ID = 317162216833155072
prediction_engine = PredictionEngine()
chart_generator = ChartGenerator()
news_analyzer = NewsAnalyzer()

@bot.event
async def on_ready():
    print(f'Bot logged in as {bot.user}')
    print(f'Bot ID: {bot.user.id}')
    
    try:
        synced = await bot.tree.sync()
        print(f'Synced {len(synced)} command(s)')
    except Exception as e:
        print(f'Failed to sync commands: {e}')
    
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="/help"))

@bot.tree.command(name="help", description="Display all available commands and usage information")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(
        title="📊 Stock Predictor Bot - Commands",
        description="Get accurate stock price predictions for short and long-term horizons.",
        color=discord.Color.blue()
    )
    embed.add_field(
        name="`/help`",
        value="Display this help message",
        inline=False
    )
    embed.add_field(
        name="`/predict <ticker>`",
        value="Get short-term (1-5 days) and long-term (6-12 months) price predictions for a stock with chart.\nExample: `/predict AAPL`",
        inline=False
    )
    embed.add_field(
        name="`/news <ticker>`",
        value="Get recent news articles for a stock with sentiment analysis.\nExample: `/news AAPL`",
        inline=False
    )
    embed.set_footer(text="Powered by advanced ML models and fundamental analysis")
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="predict", description="Get accurate stock price predictions with charts for short-term (1-5 days) and long-term (6-12 months)")
@app_commands.describe(ticker="Stock ticker symbol (e.g., AAPL, TSLA, MSFT)")
async def predict_command(interaction: discord.Interaction, ticker: str):
    ticker = ticker.upper().strip()
    
    await interaction.response.defer()
    await interaction.followup.send(f"🔍 Analyzing {ticker}... This may take a moment.")
    
    try:
        predictions = await prediction_engine.get_predictions(ticker)
        
        loop = asyncio.get_event_loop()
        
        chart_task = loop.run_in_executor(None, chart_generator.generate_chart, ticker)
        
        chart_buf = await chart_task
        
        display_ticker = predictions.get('ticker', ticker)
        
        # First embed: Current price, analysis date, and chart
        embed1 = discord.Embed(
            title=f"📈 {display_ticker} - Current Info",
            color=discord.Color.blue()
        )
        
        embed1.add_field(
            name="📊 Current Price",
            value=f"${predictions['current_price']:.2f}",
            inline=True
        )
        
        embed1.add_field(
            name="📅 Analysis Date",
            value=predictions['analysis_date'],
            inline=True
        )
        
        embed1.add_field(
            name="\u200b",
            value="\u200b",
            inline=True
        )
        
        files = []
        if chart_buf and not isinstance(chart_buf, Exception):
            chart_file = discord.File(chart_buf, filename=f"{ticker}_chart.png")
            files.append(chart_file)
            embed1.set_image(url=f"attachment://{ticker}_chart.png")
        
        await interaction.followup.send(embed=embed1, files=files if files else None)
        
        # Second embed: Predictions
        embed2 = discord.Embed(
            title=f"📊 Predictions for {display_ticker}",
            color=discord.Color.blue()
        )
        
        embed2.add_field(
            name="⚡ Short-Term Prediction (1-5 days)",
            value=f"**Predicted Price:** ${predictions['short_term']['predicted_price']:.2f}\n"
                  f"**Change:** {predictions['short_term']['change_percent']:+.2f}%\n"
                  f"**Confidence:** {predictions['short_term']['confidence']:.1f}%\n"
                  f"**Key Factors:** {predictions['short_term']['key_factors']}",
            inline=False
        )
        
        embed2.add_field(
            name="🎯 Long-Term Prediction (6-12 months)",
            value=f"**Predicted Price:** ${predictions['long_term']['predicted_price']:.2f}\n"
                  f"**Change:** {predictions['long_term']['change_percent']:+.2f}%\n"
                  f"**Confidence:** {predictions['long_term']['confidence']:.1f}%\n"
                  f"**Key Factors:** {predictions['long_term']['key_factors']}",
            inline=False
        )
        
        if predictions.get('warnings'):
            embed2.add_field(
                name="⚠️ Warnings",
                value="\n".join(predictions['warnings']),
                inline=False
            )
        
        embed2.set_footer(text="Predictions are for informational purposes only. Not financial advice.")
        
        await interaction.followup.send(embed=embed2)
        
    except Exception as e:
        embed = discord.Embed(
            title="❌ Error",
            description=f"Failed to get predictions for {ticker}.\nError: {str(e)}",
            color=discord.Color.red()
        )
        await interaction.followup.send(embed=embed)

@bot.tree.command(name="news", description="Get recent news articles for a stock with bullish/bearish sentiment analysis")
@app_commands.describe(ticker="Stock ticker symbol (e.g., AAPL, TSLA, MSFT)")
async def news_command(interaction: discord.Interaction, ticker: str):
    ticker = ticker.upper().strip()
    
    await interaction.response.defer()
    await interaction.followup.send(f"📰 Fetching news for {ticker}...")
    
    try:
        loop = asyncio.get_event_loop()
        articles = await loop.run_in_executor(None, news_analyzer.get_news, ticker, 6)
        
        if not articles:
            embed = discord.Embed(
                title=f"📰 News for {ticker}",
                description="No recent news articles found for this ticker.",
                color=discord.Color.orange()
            )
            await interaction.followup.send(embed=embed)
            return
        
        bullish_articles = [a for a in articles if a['sentiment'] == 'bullish']
        bearish_articles = [a for a in articles if a['sentiment'] == 'bearish']
        neutral_articles = [a for a in articles if a['sentiment'] == 'neutral']
        
        embed = discord.Embed(
            title=f"📰 Recent News for {ticker}",
            description=f"Found {len(articles)} articles",
            color=discord.Color.blue()
        )
        
        if bullish_articles:
            bullish_text = news_analyzer.format_news_for_embed(bullish_articles)
            embed.add_field(
                name=f"🟢 Bullish News ({len(bullish_articles)})",
                value=bullish_text[:1024],
                inline=False
            )
        
        if bearish_articles:
            bearish_text = news_analyzer.format_news_for_embed(bearish_articles)
            embed.add_field(
                name=f"🔴 Bearish News ({len(bearish_articles)})",
                value=bearish_text[:1024],
                inline=False
            )
        
        if neutral_articles and not bullish_articles and not bearish_articles:
            neutral_text = news_analyzer.format_news_for_embed(neutral_articles)
            embed.add_field(
                name=f"⚪ Recent News ({len(neutral_articles)})",
                value=neutral_text[:1024],
                inline=False
            )
        
        if neutral_articles:
            if not bullish_articles and not bearish_articles:
                neutral_text = news_analyzer.format_news_for_embed(neutral_articles)
                embed.add_field(
                    name=f"⚪ Recent News ({len(neutral_articles)})",
                    value=neutral_text[:1024],
                    inline=False
                )
            else:
                neutral_text = news_analyzer.format_news_for_embed(neutral_articles)
                embed.add_field(
                    name=f"⚪ Neutral News ({len(neutral_articles)})",
                    value=neutral_text[:1024],
                    inline=False
                )
        
        sentiment_summary = f"🟢 {len(bullish_articles)} bullish | 🔴 {len(bearish_articles)} bearish | ⚪ {len(neutral_articles)} neutral"
        embed.set_footer(text=sentiment_summary)
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        embed = discord.Embed(
            title="❌ Error",
            description=f"Failed to fetch news for {ticker}.\nError: {str(e)}",
            color=discord.Color.red()
        )
        await interaction.followup.send(embed=embed)

@bot.command(name='rules')
async def rules_command(ctx):
    if ctx.author.id != ADMIN_USER_ID:
        return
    
    embed = discord.Embed(
        title="Server Rules",
        description="Please follow the rules below:",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="1. Be Respectful",
        value="Treat all members with respect. No harassment, hate speech, or personal attacks.",
        inline=False
    )
    
    embed.add_field(
        name="2. No Spam",
        value="Avoid spamming commands or messages. Use the bot responsibly.",
        inline=False
    )
    
    embed.add_field(
        name="3. No Financial Advice",
        value="Remember that predictions are for informational purposes only. Do not treat them as financial advice.",
        inline=False
    )
    
    embed.add_field(
        name="4. Follow Discord ToS",
        value="Adhere to Discord's Terms of Service and Community Guidelines.",
        inline=False
    )
    
    embed.set_footer(text="Violations may result in warnings or removal from the server.")
    
    await ctx.send(embed=embed)

@bot.command(name='info')
async def info_command(ctx):
    if ctx.author.id != ADMIN_USER_ID:
        return
    
    embed = discord.Embed(
        title="📊 Stock Predictor Bot - Information",
        description="An advanced AI-powered stock prediction bot using machine learning and fundamental analysis.",
        color=discord.Color.green()
    )
    
    embed.add_field(
        name="🤖 How It Works",
        value="The bot uses a sophisticated ensemble of prediction models:\n"
              "• **Short-term (1-5 days)**: Technical indicators, momentum analysis, market sentiment, and price action patterns\n"
              "• **Long-term (6-12 months)**: Fundamental analysis, DCF models, financial ratios, and valuation metrics",
        inline=False
    )
    
    embed.add_field(
        name="📈 Commands",
        value="`/help` - Display all available commands\n"
              "`/predict <ticker>` - Get price predictions with charts\n"
              "`/news <ticker>` - View recent news with sentiment analysis",
        inline=False
    )
    
    embed.add_field(
        name="💡 How to Use",
        value="1. Use `/predict AAPL` to get predictions for any stock\n"
              "2. View the chart and current price in the first embed\n"
              "3. Check short-term and long-term predictions in the second embed\n"
              "4. Use `/news AAPL` to see recent news articles with sentiment",
        inline=False
    )
    
    embed.add_field(
        name="⚠️ Important Disclaimer",
        value="All predictions are for informational purposes only and should not be considered as financial advice. "
              "Always do your own research and consult with a financial advisor before making investment decisions.",
        inline=False
    )
    
    embed.add_field(
        name="🔧 Technical Details",
        value="• Uses 5+ technical indicators for short-term predictions\n"
              "• Implements 8+ fundamental models for long-term analysis\n"
              "• Real-time data from Yahoo Finance\n"
              "• Advanced sentiment analysis for news articles",
        inline=False
    )
    
    embed.set_footer(text="Powered by advanced ML models and fundamental analysis")
    
    await ctx.send(embed=embed)

@bot.command(name='cl1')
async def changelog_v1(ctx):
    if ctx.author.id != ADMIN_USER_ID:
        return
    
    embed = discord.Embed(
        title="Stock Predictor v1.0",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="\u200b",
        value="• Added `/predict` command for short-term and long-term stock price predictions\n"
              "• Added `/news` command to view recent stock news with sentiment analysis\n"
              "• Added `/help` command to display all available commands\n"
              "• Implemented ML models and fundamental analysis for accurate predictions\n"
              "• Integrated Yahoo Finance API for real-time stock data and news",
        inline=False
    )
    
    # Set footer with author name and icon (at the bottom of the embed)
    embed.set_footer(text="brian", icon_url="https://imgur.com/a/W3of0Zw")
    
    await ctx.send(embed=embed)

@bot.command(name='invite')
async def invite_command(ctx):
    if ctx.author.id != ADMIN_USER_ID:
        return
    
    embed = discord.Embed(
        title="Invite Stock Predictor Bot",
        description="Invite this bot to your server!",
        color=discord.Color.green()
    )
    
    embed.add_field(
        name="🔗 Invite Link",
        value="[Click here to invite the bot](https://discord.com/oauth2/authorize?client_id=1443395790620987554&permissions=2147862528&integration_type=0&scope=bot+applications.commands)",
        inline=False
    )
    
    await ctx.send(embed=embed)

if __name__ == "__main__":
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        print("Error: DISCORD_BOT_TOKEN not found in environment variables")
        print("For local development: Create a .env file with your Discord bot token")
        print("For Railway: Add DISCORD_BOT_TOKEN as an environment variable in Railway dashboard")
        print("  Go to your project → Variables → Add: DISCORD_BOT_TOKEN = your_token_here")
        print(f"\nDebug: All env vars starting with 'DISCORD': {[k for k in os.environ.keys() if 'DISCORD' in k.upper()]}")
    else:
        print(f"✅ Found DISCORD_BOT_TOKEN (length: {len(token)})")
        print("Starting bot...")
        bot.run(token)

