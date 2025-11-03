import { useState, type KeyboardEvent } from "react";
import "./App.css";

interface Message {
    sender: "user" | "bot";
    text: string;
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
            const res = await fetch("http://127.0.0.1:8000/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt: input }),
            });

            if (!res.ok) throw new Error("Ошибка сети");

            const data: { response: string } = await res.json();
            const botMessage: Message = { sender: "bot", text: data.response };
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
