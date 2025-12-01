import { GoogleGenAI, Type } from "@google/genai";
import { User, Title, WatchHistory, ChurnPrediction, Recommendation } from '../types';

const getClient = () => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("API_KEY is not defined");
  }
  return new GoogleGenAI({ apiKey });
};

export const analyzeChurnRisk = async (user: User, history: WatchHistory[]): Promise<ChurnPrediction> => {
  const ai = getClient();
  const modelId = "gemini-2.5-flash";

  const prompt = `
    Analyze the churn risk for the following OTT user.
    User Profile: ${JSON.stringify(user)}
    Recent Activity Stats: ${history.length} titles watched, Average rating: ${history.reduce((acc, h) => acc + h.rating, 0) / (history.length || 1)}.

    Return a JSON object with:
    - probability: number (0-100)
    - riskLevel: "Low", "Medium", or "High"
    - factors: array of strings (top 3 reasons)
    - retentionStrategy: string (one specific actionable strategy)
  `;

  try {
    const response = await ai.models.generateContent({
      model: modelId,
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            probability: { type: Type.NUMBER },
            riskLevel: { type: Type.STRING, enum: ["Low", "Medium", "High"] },
            factors: { type: Type.ARRAY, items: { type: Type.STRING } },
            retentionStrategy: { type: Type.STRING },
          },
          required: ["probability", "riskLevel", "factors", "retentionStrategy"]
        }
      }
    });

    return JSON.parse(response.text) as ChurnPrediction;
  } catch (error) {
    console.error("Gemini Churn Error:", error);
    return {
      probability: 50,
      riskLevel: "Medium",
      factors: ["Error analyzing data", "Incomplete profile"],
      retentionStrategy: "Offer standard discount"
    };
  }
};

export const getHybridRecommendations = async (user: User, userHistory: WatchHistory[], allTitles: Title[]): Promise<Recommendation[]> => {
  const ai = getClient();
  const modelId = "gemini-2.5-flash";
  
  // Sample available titles to keep prompt size manageable
  const sampleTitles = allTitles.slice(0, 30).map(t => `${t.title_name} (${t.genre})`);
  const watchedTitles = userHistory.slice(0, 10).map(h => h.title_id); // In real app, map to names

  const prompt = `
    Act as a Hybrid Recommendation Engine.
    User Profile: Age ${user.age}, Genre Pref: ${user.preferred_genre}.
    Watched History IDs: ${JSON.stringify(watchedTitles)}.
    
    Available Catalog Sample: ${JSON.stringify(sampleTitles)}.
    
    Recommend 3 titles from the catalog that fit the user's profile and history.
    Focus on Content-Based (genre match) and Collaborative patterns (implied).

    Return JSON.
  `;

  try {
    const response = await ai.models.generateContent({
      model: modelId,
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              title_name: { type: Type.STRING },
              match_score: { type: Type.NUMBER, description: "0-100 score" },
              reason: { type: Type.STRING }
            },
            required: ["title_name", "match_score", "reason"]
          }
        }
      }
    });

    return JSON.parse(response.text) as Recommendation[];
  } catch (error) {
    console.error("Gemini Recs Error:", error);
    return [];
  }
};

export const getGeneralInsights = async (query: string, dataSummary: string): Promise<string> => {
  const ai = getClient();
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: `
        You are a Senior Data Analyst for a Streaming Service.
        Context Data: ${dataSummary}
        
        User Query: "${query}"
        
        Provide a concise, professional answer. Use markdown for formatting.
      `
    });
    return response.text;
  } catch (e) {
    return "Unable to generate insights at this moment.";
  }
};