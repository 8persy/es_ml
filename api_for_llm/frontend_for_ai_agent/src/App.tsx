import { useState, type KeyboardEvent } from "react";
import "./App.css";

interface Message {
    sender: "user" | "bot";
    text: string;
}

interface NewsArticle {
    title: string;
    url: string;
    source: string;
}

interface ApiResponse {
    keyword?: string;
    news?: NewsArticle[];
    error?: string;
}

function App() {
    const [input, setInput] = useState<string>("");
    const [messages, setMessages] = useState<Message[]>([]);
    const [loading, setLoading] = useState<boolean>(false);

    const sendMessage = async () => {
        if (!input.trim()) return;
        setLoading(true);

        const userMessage: Message = { sender: "user", text: input };
        setMessages((prev) => [...prev, userMessage]);

        try {
            const res = await fetch("http://127.0.0.1:8080/summarize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: input }),
            });

            if (!res.ok) throw new Error("Ошибка сети");

            const data: ApiResponse = await res.json();

            let botText = "";

            if (data.error) {
                botText = `Ошибка: ${data.error}`;
            } else if (!data.news || data.news.length === 0) {
                botText = `Не удалось найти новости по теме "${data.keyword}".`;
            } else {
                botText = `Вот новости на тему: "${data.keyword}"\n\n`;
                botText += data.news
                    .map(
                        (article, idx) =>
                            `${idx + 1}. [${article.title}] (${article.url}) — ${article.source}`
                    )
                    .join("\n");
            }

            const botMessage: Message = { sender: "bot", text: botText };
            setMessages((prev) => [...prev, botMessage]);
        } catch (err) {
            console.error(err);
            setMessages((prev) => [
                ...prev,
                { sender: "bot", text: "Ошибка соединения с API 🛑" },
            ]);
        } finally {
            setInput("");
            setLoading(false);
        }
    };

    const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter") sendMessage();
    };

    return (
        <div className="chat-container">
            <div className="chat-box">
                <h1>Локальный чат с LLM</h1>

                <div className="messages">
                    {messages.map((msg, i) => (
                        <div
                            key={i}
                            className={`message ${
                                msg.sender === "user" ? "user" : "bot"
                            }`}
                            style={{ whiteSpace: "pre-line" }}
                        >
                            {msg.text}
                        </div>
                    ))}
                    {loading && <p className="thinking">🤔 Думает...</p>}
                </div>

                <div className="input-area">
                    <input
                        type="text"
                        placeholder="Введите сообщение..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyPress}
                    />
                    <button onClick={sendMessage} disabled={loading}>
                        Отправить
                    </button>
                </div>
            </div>
        </div>
    );
}

export default App;
