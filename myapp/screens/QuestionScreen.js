import React, { useState, useEffect } from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';
import api from '../api';

export default function QuestionScreen({ route, navigation }) {
  const { userId, firstQuestion } = route.params;
  const [currentQuestion, setCurrentQuestion] = useState(firstQuestion);
  const [answers, setAnswers] = useState([]);

  const handleAnswer = async (answer) => {
    try {
      const newAnswers = [...answers, {
        questionId: currentQuestion.id,
        answer
      }];
      
      const response = await api.submitAnswer({
        userId,
        questionId: currentQuestion.id,
        answer
      });

      if (response.completed) {
        navigation.navigate('Results', { userId });
      } else {
        setCurrentQuestion(response.nextQuestion);
        setAnswers(newAnswers);
      }
    } catch (error) {
      console.error('Ошибка:', error);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.questionText}>{currentQuestion.text}</Text>
      
      <View style={styles.buttonsContainer}>
        <Button 
          title="Да" 
          onPress={() => handleAnswer('yes')}
          color="#2ecc71"
        />
        <Button 
          title="Иногда" 
          onPress={() => handleAnswer('sometimes')}
          color="#f39c12"
        />
        <Button 
          title="Нет" 
          onPress={() => handleAnswer('no')}
          color="#e74c3c"
        />
      </View>

      <Text style={styles.progress}>
        Вопрос {currentQuestion.number} из {currentQuestion.total}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
  },
  questionText: {
    fontSize: 18,
    marginBottom: 30,
    textAlign: 'center',
  },
  buttonsContainer: {
    gap: 15,
    marginBottom: 30,
  },
  progress: {
    textAlign: 'center',
    color: '#7f8c8d',
  },
});