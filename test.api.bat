REM Basic suggestions
curl -X GET "http://localhost:8000/api/v1/suggestions" -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNzU2MzM1NzcxfQ.IlqcCen3n_Oyyho13AITnMTLHJK4I5qcHb75RlPrQyrQ"

REM Filter by Discovery category
curl -X GET "http://localhost:8000/api/v1/suggestions?category_filter=Discovery" -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNzU2MzM1NzcxfQ.IlqcCen3n_Oyyho13AITnMTLHJK4I5qcHb75RlPrQyrQ"

REM Filter by high urgency
curl -X GET "http://localhost:8000/api/v1/suggestions?urgency_filter=high" -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNzU2MzM1NzcxfQ.IlqcCen3n_Oyyho13AITnMTLHJK4I5qcHb75RlPrQyrQ"

REM Combined filters
curl -X GET "http://localhost:8000/api/v1/suggestions?category_filter=Performance&urgency_filter=medium" -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNzU2MzM1NzcxfQ.IlqcCen3n_Oyyho13AITnMTLHJK4I5qcHb75RlPrQyrQ"

REM Execute agent query
curl -X POST "http://localhost:8000/api/v1/suggestions/execute" -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNzU2MzM1NzcxfQ.IlqcCen3n_Oyyho13AITnMTLHJK4I5qcHb75RlPrQyrQ" -H "Content-Type: application/json" -d "{\"id\":\"poor_performance\",\"type\":\"agent_query\",\"query\":\"My portfolio's Sharpe ratio is 0.00. How can I improve risk-adjusted returns?\",\"target_agent\":\"quantitative\"}"

REM Execute workflow trigger
curl -X POST "http://localhost:8000/api/v1/suggestions/execute" -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNzU2MzM1NzcxfQ.IlqcCen3n_Oyyho13AITnMTLHJK4I5qcHb75RlPrQyrQ" -H "Content-Type: application/json" -d "{\"id\":\"sector_rotation\",\"type\":\"workflow_trigger\",\"query\":\"Analyze current sector rotation trends.\",\"workflow_type\":\"sector_analysis\"}"

REM Submit feedback
curl -X POST "http://localhost:8000/api/v1/suggestions/feedback" -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNzU2MzM1NzcxfQ.IlqcCen3n_Oyyho13AITnMTLHJK4I5qcHb75RlPrQyrQ" -H "Content-Type: application/json" -d "{\"suggestion_id\":\"poor_performance\",\"rating\":5,\"helpful\":true,\"comments\":\"This was a very relevant suggestion.\"}"

REM Get statistics
curl -X GET "http://localhost:8000/api/v1/suggestions/stats?days=30" -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNzU2MzM1NzcxfQ.IlqcCen3n_Oyyho13AITnMTLHJK4I5qcHb75RlPrQyrQ"

REM Contextual chat suggestions
curl -X POST "http://localhost:8000/api/v1/suggestions/chat/contextual" -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNzU2MzM1NzcxfQ.IlqcCen3n_Oyyho13AITnMTLHJK4I5qcHb75RlPrQyrQ" -H "Content-Type: application/json" -d "{\"current_agent\":\"quantitative\",\"last_response\":{\"text\":\"Your portfolio risk score is high.\"},\"conversation_history\":[{\"role\":\"user\",\"content\":\"analyze my risk\"}]}"

REM Dashboard widget suggestions
curl -X GET "http://localhost:8000/api/v1/suggestions/dashboard/widgets?widget_context=risk_metrics" -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNzU2MzM1NzcxfQ.IlqcCen3n_Oyyho13AITnMTLHJK4I5qcHb75RlPrQyrQ"