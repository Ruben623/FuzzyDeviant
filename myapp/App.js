import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import StartScreen from './src/screens/StartScreen';
import QuestionScreen from './src/screens/QuestionScreen';
import ResultsScreen from './src/screens/ResultsScreen';

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Start">
        <Stack.Screen 
          name="Start" 
          component={StartScreen} 
          options={{ title: 'Тест девиантного поведения' }}
        />
        <Stack.Screen 
          name="Question" 
          component={QuestionScreen} 
          options={{ title: 'Вопрос' }}
        />
        <Stack.Screen 
          name="Results" 
          component={ResultsScreen} 
          options={{ title: 'Результаты' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}