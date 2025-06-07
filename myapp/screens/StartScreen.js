import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet } from 'react-native';
import api from '../api';

export default function StartScreen({ navigation }) {
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('male');

  const handleStart = async () => {
    try {
      const response = await api.startTest({ age, gender });
      navigation.navigate('Question', {
        userId: response.userId,
        firstQuestion: response.firstQuestion
      });
    } catch (error) {
      console.error('Ошибка:', error);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Введите ваши данные</Text>
      
      <TextInput
        style={styles.input}
        placeholder="Ваш возраст"
        keyboardType="numeric"
        value={age}
        onChangeText={setAge}
      />

      <View style={styles.radioGroup}>
        <Button 
          title="Мужской" 
          onPress={() => setGender('male')}
          color={gender === 'male' ? '#3498db' : '#95a5a6'}
        />
        <Button 
          title="Женский" 
          onPress={() => setGender('female')}
          color={gender === 'female' ? '#3498db' : '#95a5a6'}
        />
      </View>

      <Button 
        title="Начать тест" 
        onPress={handleStart}
        disabled={!age}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
  },
  title: {
    fontSize: 20,
    marginBottom: 20,
    textAlign: 'center',
  },
  input: {
    height: 40,
    borderColor: '#ddd',
    borderWidth: 1,
    marginBottom: 20,
    padding: 10,
    borderRadius: 5,
  },
  radioGroup: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
  },
});