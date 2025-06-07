const API_BASE_URL = 'http://ВАШ_IP_СЕРВЕРА:5000/api';

export default {
  async startTest(userData) {
    const response = await fetch(`${API_BASE_URL}/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(userData),
    });
    return await response.json();
  },

  async submitAnswer(answerData) {
    const response = await fetch(`${API_BASE_URL}/submit_answer`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(answerData),
    });
    return await response.json();
  },

  async getResults(userId) {
    const response = await fetch(`${API_BASE_URL}/results/${userId}`);
    return await response.json();
  }
};